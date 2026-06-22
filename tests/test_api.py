"""Tests de l'API de scoring.

On teste la logique des endpoints en appelant directement les fonctions
(`health`, `predict`) plutôt que via HTTP : c'est plus robuste (indépendant de la
version du TestClient/Starlette) et ça couvre exactement le même code.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from mapie.classification import SplitConformalClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from readmission_risk.common.config import settings
from readmission_risk.modeling.calibrate import DataFrameAdapter
from readmission_risk.modeling.gboost import build_pipeline
from readmission_risk.serving import api


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    df = pd.DataFrame([
        {"num1": (x := rng.normal()), "cat1": str(rng.choice(["A", "B", "C"])), "y": int(x > 0.3)}
        for _ in range(400)
    ])
    cols = ["num1", "cat1"]
    spw = float((df["y"] == 0).sum() / max((df["y"] == 1).sum(), 1))

    base = build_pipeline(["num1"], ["cat1"], {"n_estimators": 40}, spw).fit(df[cols], df["y"])
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(df[cols], df["y"])
    scc = SplitConformalClassifier(
        estimator=DataFrameAdapter(cal, cols),
        confidence_level=0.9, conformity_score="lac", prefit=True,
    )
    scc.conformalize(df[cols], df["y"])

    joblib.dump(
        {"model": cal, "conformal": scc, "base_pipeline": base, "feature_cols": cols,
         "confidence_level": 0.9, "calibration_method": "sigmoid"},
        tmp_path / "model.joblib",
    )
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "model_filename", "model.joblib")
    api._STATE.clear()
    yield
    api._STATE.clear()


def test_health(loaded):
    out = api.health()
    assert out["status"] == "ok" and out["model_loaded"] is True


def test_predict_structure(loaded):
    resp = api.predict(api.PredictRequest(features={"num1": 1.5, "cat1": "A"}))
    assert 0.0 <= resp.risk <= 1.0
    assert resp.risk_label in {"non_readmis", "readmission_30j"}
    assert len(resp.prediction_set) >= 1
    assert len(resp.top_reasons) >= 1
    assert resp.confidence_level == 0.9


def test_predict_missing_features_imputed(loaded):
    resp = api.predict(api.PredictRequest(features={}))  # tout manquant -> imputé
    assert 0.0 <= resp.risk <= 1.0

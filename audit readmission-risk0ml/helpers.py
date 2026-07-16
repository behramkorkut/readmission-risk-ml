"""Helpers partagés entre fichiers de tests (ce module n'est pas collecté par pytest)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from readmission_risk.modeling.calibrate import DataFrameAdapter
from readmission_risk.modeling.gboost import build_pipeline

DEMO_FEATURE_COLS = ["num1", "cat1"]


def build_demo_model_bundle(model_path: Path, seed: int = 0) -> None:
    """Entraîne un mini-modèle (calibré + conformal) et le sérialise en joblib.

    Version jouet de la chaîne réelle (`readmission-calibrate`) pour les tests :
    2 features, 400 lignes, LightGBM 40 arbres -> s'entraîne en ~1 s.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        [{"num1": (x := rng.normal()), "cat1": str(rng.choice(["A", "B", "C"])), "y": int(x > 0.3)} for _ in range(400)]
    )
    cols = DEMO_FEATURE_COLS
    spw = float((df["y"] == 0).sum() / max((df["y"] == 1).sum(), 1))

    base = build_pipeline(["num1"], ["cat1"], {"n_estimators": 40}, spw).fit(df[cols], df["y"])
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(df[cols], df["y"])
    scc = SplitConformalClassifier(
        estimator=DataFrameAdapter(cal, cols),
        confidence_level=0.9,
        conformity_score="lac",
        prefit=True,
    )
    scc.conformalize(df[cols], df["y"])

    joblib.dump(
        {
            "model": cal,
            "conformal": scc,
            "base_pipeline": base,
            "feature_cols": cols,
            "confidence_level": 0.9,
            "calibration_method": "sigmoid",
        },
        model_path,
    )

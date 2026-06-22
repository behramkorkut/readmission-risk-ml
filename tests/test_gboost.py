"""Tests du gradient boosting + tuning (sur mini-jeu synthétique, rapide)."""

import numpy as np
import pandas as pd

from readmission_risk.modeling.gboost import build_pipeline, tune


def _synthetic(n_patients=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        s = rng.normal()
        for _ in range(rng.integers(1, 4)):
            rows.append({
                "num1": s + rng.normal(0, 0.5),
                "cat1": rng.choice(["A", "B", "C"]),
                "patient_nbr": pid,
                "y": int((s + rng.normal(0, 0.5)) > 0.7),
            })
    return pd.DataFrame(rows)


def test_build_pipeline_has_lgbm():
    pipe = build_pipeline(["num1"], ["cat1"], {"n_estimators": 50}, scale_pos_weight=3.0)
    assert pipe.named_steps["clf"].__class__.__name__ == "LGBMClassifier"


def test_tune_runs_and_returns_params():
    df = _synthetic()
    X = df[["num1", "cat1"]]
    study = tune(
        X, df["y"], df["patient_nbr"],
        numeric=["num1"], categorical=["cat1"],
        scale_pos_weight=3.0, n_trials=2, folds=2, seed=42,
    )
    assert "n_estimators" in study.best_params
    assert 0.0 <= study.best_value <= 1.0  # PR-AUC valide

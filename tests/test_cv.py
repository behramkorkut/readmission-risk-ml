"""Tests de la validation croisée groupée."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from readmission_risk.modeling.cv import grouped_cv_scores, summarize


def _synthetic(n_patients=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        signal = rng.normal()
        for _ in range(rng.integers(1, 4)):
            x = signal + rng.normal(0, 0.5)
            y = int((signal + rng.normal(0, 0.5)) > 0.7)  # ~déséquilibré
            rows.append({"x1": x, "x2": rng.normal(), "patient": pid, "y": y})
    return pd.DataFrame(rows)


def test_grouped_cv_returns_expected_keys_and_lengths():
    df = _synthetic()
    scores = grouped_cv_scores(
        LogisticRegression(max_iter=200),
        df[["x1", "x2"]], df["y"], groups=df["patient"], n_splits=3, seed=42,
    )
    for key in ("test_pr_auc", "test_roc_auc", "test_neg_brier"):
        assert key in scores
        assert len(scores[key]) == 3  # un score par fold


def test_summarize_keys():
    df = _synthetic()
    scores = grouped_cv_scores(
        LogisticRegression(max_iter=200),
        df[["x1", "x2"]], df["y"], groups=df["patient"], n_splits=3, seed=42,
    )
    summary = summarize(scores)
    assert {"pr_auc_mean", "roc_auc_mean", "brier_mean"} <= set(summary)
    assert 0.0 <= summary["pr_auc_mean"] <= 1.0
    assert summary["brier_mean"] >= 0.0  # Brier repassé en positif

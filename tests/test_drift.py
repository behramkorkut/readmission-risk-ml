"""Tests du monitoring de dérive (injection + décision ; le rapport Evidently est testé à part)."""

import numpy as np
import pandas as pd

from readmission_risk.monitoring.drift import (
    _extract_drift,
    inject_drift,
    retraining_decision,
)


def _df(n=500):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "time_in_hospital": rng.integers(1, 10, n),
        "num_medications": rng.integers(1, 20, n),
        "number_inpatient": rng.integers(0, 3, n),
        "age": rng.choice(["[40-50)", "[50-60)", "[60-70)"], n),
        "medical_specialty": rng.choice(["Cardiology", "Surgery"], n),
    })


def test_inject_drift_shifts_distributions():
    df = _df()
    drifted = inject_drift(df, seed=1)
    # population plus "lourde" : plus de médicaments en moyenne
    assert drifted["num_medications"].mean() > df["num_medications"].mean()
    # davantage de manquants sur medical_specialty
    assert drifted["medical_specialty"].isna().mean() > df["medical_specialty"].isna().mean()
    # bornes respectées
    assert drifted["time_in_hospital"].between(1, 14).all()


def test_retraining_decision():
    trig, _ = retraining_decision(0.5, threshold=0.3)
    assert trig is True
    notrig, _ = retraining_decision(0.1, threshold=0.3)
    assert notrig is False


def test_extract_drift():
    fake = {"metrics": [{"value": {"count": 4.0, "share": 0.4}}, {"value": 0.01}]}
    share, count = _extract_drift(fake)
    assert share == 0.4 and count == 4

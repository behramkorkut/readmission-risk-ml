"""Tests du split par patient (anti-fuite)."""

import numpy as np
import pandas as pd
import pytest

from readmission_risk.data.split import assert_no_group_overlap, make_holdout_split


def _synthetic(n_patients=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        for _ in range(rng.integers(1, 4)):  # 1 à 3 séjours par patient
            rows.append({"patient_nbr": pid, "y": int(rng.random() < 0.2)})
    return pd.DataFrame(rows)


def test_no_patient_overlap():
    df = _synthetic()
    train, test = make_holdout_split(df, "y", "patient_nbr", test_size=0.2, seed=42)
    overlap = set(train["patient_nbr"]) & set(test["patient_nbr"])
    assert overlap == set()  # aucun patient partagé


def test_split_sizes_approx():
    df = _synthetic()
    train, test = make_holdout_split(df, "y", "patient_nbr", test_size=0.2, seed=42)
    ratio = len(test) / len(df)
    assert 0.12 < ratio < 0.28  # ~20 %, avec tolérance (regroupement par patient)


def test_target_present_in_both():
    df = _synthetic()
    train, test = make_holdout_split(df, "y", "patient_nbr", test_size=0.2, seed=42)
    assert train["y"].nunique() == 2 and test["y"].nunique() == 2


def test_assert_no_group_overlap_raises():
    a = pd.DataFrame({"patient_nbr": [1, 2]})
    b = pd.DataFrame({"patient_nbr": [2, 3]})  # patient 2 partagé
    with pytest.raises(AssertionError):
        assert_no_group_overlap(a, b, "patient_nbr")

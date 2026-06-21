"""Tests du nettoyage et de l'anti-fuite."""

import numpy as np
import pandas as pd

from readmission_risk.data.clean import (
    _icd9_group,
    add_binary_target,
    clean,
    drop_leakage_encounters,
)


def test_binary_target():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
    out = add_binary_target(df)
    assert out["readmitted_30d"].tolist() == [1, 0, 0]
    assert "readmitted" not in out.columns  # cible 3-classes retirée (anti-fuite)


def test_drop_leakage_encounters():
    df = pd.DataFrame({"discharge_disposition_id": [1, 11, 13, 3, 14]})
    out = drop_leakage_encounters(df)
    # 11, 13, 14 = décès/hospice -> retirés ; restent 1 et 3.
    assert out["discharge_disposition_id"].tolist() == [1, 3]


def test_icd9_grouping():
    assert _icd9_group("250.83") == "Diabetes"
    assert _icd9_group("428") == "Circulatory"
    assert _icd9_group("486") == "Respiratory"
    assert _icd9_group("V45") == "Other"
    assert _icd9_group("E909") == "Other"
    assert _icd9_group(np.nan) == "Missing"


def test_clean_end_to_end():
    df = pd.DataFrame({
        "discharge_disposition_id": [1, 11],          # 2e = décès -> retiré
        "gender": ["Male", "Female"],
        "diag_1": ["250.5", "428"],
        "diag_2": ["?", "486"],
        "diag_3": ["V45", "?"],
        "weight": ["?", "[50-75)"],
        "readmitted": ["<30", "NO"],
    })
    out = clean(df)
    assert len(out) == 1                       # le séjour décès est retiré
    assert out.loc[0, "readmitted_30d"] == 1
    assert "weight" not in out.columns
    assert "diag_1" not in out.columns and "diag_1_group" in out.columns
    assert out.loc[0, "diag_1_group"] == "Diabetes"
    assert out.loc[0, "diag_2_group"] == "Missing"  # « ? » -> NaN -> Missing

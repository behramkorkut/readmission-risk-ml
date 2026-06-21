"""Tests du pipeline de features."""

import numpy as np
import pandas as pd

from readmission_risk.features.build import build_feature_spec, build_preprocessor


def _df():
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4],
            "patient_nbr": [10, 20, 30, 40],
            "time_in_hospital": [3, 5, 2, 8],
            "race": ["Caucasian", "Asian", None, "Caucasian"],
            "admission_type_id": [1, 1, 2, 3],   # code entier -> catégoriel
            "examide": ["No", "No", "No", "No"],  # constante -> écartée
            "readmitted_30d": [0, 1, 0, 1],
        }
    )


def test_spec_exclut_ids_cible_et_constantes():
    numeric, categorical, constant = build_feature_spec(_df(), "readmitted_30d")
    assert numeric == ["time_in_hospital"]
    assert "race" in categorical and "admission_type_id" in categorical
    assert "encounter_id" not in categorical and "patient_nbr" not in categorical
    assert constant == ["examide"]  # médicament jamais prescrit


def test_preprocessor_fit_transform():
    df = _df()
    numeric, categorical, _ = build_feature_spec(df, "readmitted_30d")
    pre = build_preprocessor(numeric, categorical)
    X = pre.fit_transform(df[numeric + categorical])
    arr = X.toarray() if hasattr(X, "toarray") else X
    assert arr.shape[0] == 4
    assert not np.isnan(arr).any()  # imputation -> aucun NaN en sortie


def test_categorie_inconnue_en_test_ne_casse_pas():
    df = _df()
    numeric, categorical, _ = build_feature_spec(df, "readmitted_30d")
    pre = build_preprocessor(numeric, categorical)
    pre.fit(df[numeric + categorical])

    # 'race' = 'Hispanic' jamais vu à l'entraînement -> doit être ignoré sans erreur.
    new = df.copy()
    new.loc[0, "race"] = "Hispanic"
    X = pre.transform(new[numeric + categorical])
    assert X.shape[1] == pre.transform(df[numeric + categorical]).shape[1]

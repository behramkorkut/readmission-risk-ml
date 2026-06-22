"""Tests de l'audit d'équité (le calcul SHAP est trop lourd pour un test unitaire)."""

import numpy as np
import pandas as pd

from readmission_risk.evaluation.explain import subgroup_metrics


def test_subgroup_metrics_structure_et_valeurs():
    df = pd.DataFrame({
        "gender": ["M", "M", "F", "F", "F", "M"],
        "readmitted_30d": [1, 0, 1, 0, 1, 0],
    })
    proba = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.1])
    out = subgroup_metrics(df, "readmitted_30d", proba, ["gender"])

    assert set(out["groupe"]) == {"M", "F"}
    # Effectifs corrects par groupe
    assert out.loc[out["groupe"] == "F", "n"].iloc[0] == 3
    # Taux réel F = 2/3
    assert abs(out.loc[out["groupe"] == "F", "taux_reel"].iloc[0] - 0.667) < 0.01
    # ROC-AUC présent (les deux classes existent par groupe)
    assert out["roc_auc"].notna().all()


def test_subgroup_metrics_auc_nan_si_une_seule_classe():
    df = pd.DataFrame({"g": ["A", "A"], "y": [1, 1]})  # une seule classe -> AUC indéfinie
    out = subgroup_metrics(df, "y", np.array([0.6, 0.7]), ["g"])
    assert out["roc_auc"].iloc[0] is None

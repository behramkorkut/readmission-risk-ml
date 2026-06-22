"""Validation croisée groupée par patient (le cœur de l'évaluation honnête).

On encapsule `cross_validate` avec StratifiedGroupKFold : le pipeline complet
(préprocesseur + modèle) est ré-ajusté dans chaque fold, sur le train du fold
uniquement → aucune fuite. Les groupes (patient_nbr) ne se chevauchent pas
entre train et test du fold.

Métriques adaptées au déséquilibre :
- pr_auc (average precision) : métrique PRIMAIRE quand les positifs sont rares.
- roc_auc : capacité de discrimination globale (à lire avec prudence si déséquilibre).
- neg_brier : qualité de la CALIBRATION des probabilités (plus proche de 0 = mieux).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

# Warning bénin : la matrice creuse issue du ColumnTransformer n'a pas de noms de
# colonnes, ce que LightGBM signale à la prédiction. Aucun impact -> on le tait.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "neg_brier": "neg_brier_score",
}


def grouped_cv_scores(
    estimator: BaseEstimator,
    X,
    y,
    groups,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Lance la CV groupée et renvoie le dict de scores (un tableau par métrique)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_validate(
        estimator, X, y,
        groups=groups,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )


def summarize(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Résume les scores de CV en moyennes/écarts-types lisibles."""
    pr, roc = scores["test_pr_auc"], scores["test_roc_auc"]
    brier = -scores["test_neg_brier"]  # on repasse en Brier positif (plus bas = mieux)
    return {
        "pr_auc_mean": float(pr.mean()),
        "pr_auc_std": float(pr.std()),
        "roc_auc_mean": float(roc.mean()),
        "roc_auc_std": float(roc.std()),
        "brier_mean": float(brier.mean()),
        "brier_std": float(brier.std()),
    }

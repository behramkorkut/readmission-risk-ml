"""Split train/test PAR PATIENT (anti-fuite #1).

Un même patient peut avoir plusieurs séjours. Si certains de ses séjours sont en
train et d'autres en test, le modèle « connaît » déjà le patient → fuite, et les
scores de test sont trompeusement optimistes. On regroupe donc par `patient_nbr`.

On utilise StratifiedGroupKFold : il garantit que (1) les groupes (patients) ne
se chevauchent pas entre folds, et (2) la proportion de la classe positive est
préservée — important vu le fort déséquilibre (~11 %).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_holdout_split(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie (train, test) patient-disjoints et stratifiés sur la cible.

    test_size=0.2 -> 5 folds, on prend le premier comme test (~20 %).
    """
    n_splits = round(1 / test_size)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[target_col], groups=df[group_col]))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return train, test


def assert_no_group_overlap(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> None:
    """Garde-fou : lève une erreur si un même patient est dans train ET test."""
    overlap = set(train[group_col]) & set(test[group_col])
    if overlap:
        raise AssertionError(f"Fuite patient ! {len(overlap)} patients dans train ET test.")

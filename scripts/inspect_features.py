"""Construit le préprocesseur sur le TRAIN et l'applique au TEST (vérification).

Usage : uv run python scripts/inspect_features.py
"""

from __future__ import annotations

import numpy as np

from readmission_risk.common.config import settings
from readmission_risk.data.loaders import load_clean
from readmission_risk.data.split import make_holdout_split
from readmission_risk.features.build import build_feature_spec, build_preprocessor


def main() -> None:
    df = load_clean()
    train, test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )

    numeric, categorical, constant = build_feature_spec(train, settings.target_col)
    print(f"Numériques ({len(numeric)}) : {numeric}")
    print(f"Catégorielles ({len(categorical)}) : {categorical}")
    print(f"Constantes écartées ({len(constant)}) : {constant}\n")

    pre = build_preprocessor(numeric, categorical)
    cols = numeric + categorical

    # Ajustement SUR LE TRAIN UNIQUEMENT, puis transformation du test.
    X_train = pre.fit_transform(train[cols])
    X_test = pre.transform(test[cols])

    print(f"Dimension après encodage : {X_train.shape[1]} features")
    print(f"Train transformé : {X_train.shape} | Test transformé : {X_test.shape}")
    print(f"Mêmes colonnes train/test : {X_train.shape[1] == X_test.shape[1]} (handle_unknown OK)")

    # Sortie creuse (sparse) : on vérifie l'absence de NaN sur les valeurs stockées.
    has_nan = bool(np.isnan(X_train.data).any()) if hasattr(X_train, "data") else bool(np.isnan(X_train).any())
    print(f"NaN dans la sortie : {has_nan} (doit être False)")


if __name__ == "__main__":
    main()

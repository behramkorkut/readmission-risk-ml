"""Inspecte le résultat du nettoyage et vérifie le split par patient.

Usage : uv run python scripts/inspect_clean.py
(nécessite d'avoir lancé `readmission-clean` au préalable)
"""

from __future__ import annotations

from readmission_risk.common.config import settings
from readmission_risk.data.loaders import load_clean, load_raw
from readmission_risk.data.split import assert_no_group_overlap, make_holdout_split


def main() -> None:
    raw = load_raw()
    df = load_clean()

    print("=== Nettoyage ===")
    print(f"Séjours bruts    : {len(raw)}")
    print(f"Séjours nettoyés : {len(df)}  (retirés : {len(raw) - len(df)} — décès/hospice + genre invalide)")
    print(f"Colonnes : {raw.shape[1]} -> {df.shape[1]} (diag bruts regroupés, weight/readmitted retirés)")
    print(f"Taux de réadmission <30j : {df[settings.target_col].mean():.1%}\n")

    print("=== Catégories ICD-9 (diag_1_group) ===")
    print(df["diag_1_group"].value_counts(dropna=False).to_string())
    print()

    print("=== Split par patient (hold-out) ===")
    train, test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )
    assert_no_group_overlap(train, test, settings.patient_id_col)  # garde-fou
    print(f"Train : {len(train):>6} séjours | {train[settings.patient_id_col].nunique():>6} patients | "
          f"taux+ {train[settings.target_col].mean():.1%}")
    print(f"Test  : {len(test):>6} séjours | {test[settings.patient_id_col].nunique():>6} patients | "
          f"taux+ {test[settings.target_col].mean():.1%}")
    overlap = set(train[settings.patient_id_col]) & set(test[settings.patient_id_col])
    print(f"Patients communs train/test : {len(overlap)}  (doit être 0 — anti-fuite OK)")


if __name__ == "__main__":
    main()

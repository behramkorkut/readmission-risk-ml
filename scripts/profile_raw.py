"""Profilage des données brutes : structure, fuite patient, cible, manquants.

Usage : uv run python scripts/profile_raw.py
"""

from __future__ import annotations

from readmission_risk.data.loaders import load_raw
from readmission_risk.validation.schema import MISSING_TOKEN, validate_raw


def main() -> None:
    df = load_raw()

    # --- Structure & fuite patient ---
    n_rows, n_cols = df.shape
    n_patients = df["patient_nbr"].nunique()
    print(f"Shape : {n_rows} séjours x {n_cols} colonnes")
    print(f"Patients uniques : {n_patients} ({n_rows / n_patients:.2f} séjours/patient en moyenne)")
    print(f"-> {n_rows - n_patients} séjours sont des ré-occurrences : split PAR PATIENT obligatoire.\n")

    # --- Validation du contrat de données ---
    try:
        validate_raw(df)
        print("Validation Pandera : OK\n")
    except Exception as exc:  # noqa: BLE001
        print("Validation Pandera : ÉCHEC")
        fc = getattr(exc, "failure_cases", None)
        if fc is not None:
            print(fc.head(10).to_string(index=False))
        print()

    # --- Cible ---
    print("Cible 'readmitted' :")
    print(df["readmitted"].value_counts(dropna=False).to_string())
    pos = (df["readmitted"] == "<30").mean()
    print(f"-> readmitted_30d (classe positive '<30') : {pos:.1%} — fort déséquilibre.\n")

    # --- Manquants (NaN OU sentinelle '?') ---
    missing = (df.isna() | (df == MISSING_TOKEN)).mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("Top colonnes avec valeurs manquantes ('?' ou NaN) :")
    for col, frac in missing.head(12).items():
        print(f"  {col:<22} {frac:>6.1%}")


if __name__ == "__main__":
    main()

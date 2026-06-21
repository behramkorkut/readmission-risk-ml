"""Schéma de validation des données brutes (contrat de données, via Pandera).

But : poser une porte de qualité dès l'entrée du pipeline. Toute donnée qui ne
respecte pas le contrat est détectée immédiatement, plutôt que de produire des
erreurs silencieuses (ou pire, un modèle entraîné sur des données corrompues).

On valide les colonnes à forte valeur (cible, identifiants, variables clés) avec
`strict=False` : les autres colonnes restent autorisées sans contrainte stricte.
Les valeurs manquantes sont codées « ? » dans ce dataset (chaîne, pas NaN) — on
les neutralisera à l'étape de nettoyage, le schéma les tolère ici.
"""

from __future__ import annotations

try:  # pandera >= 0.20 : sous-module dédié pandas
    import pandera.pandas as pa
except ModuleNotFoundError:  # versions plus anciennes
    import pandera as pa

# Tranches d'âge bucketisées du dataset : [0-10), [10-20), ..., [90-100).
AGE_BUCKETS = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]
GENDERS = ["Male", "Female", "Unknown/Invalid"]
RACES = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "?"]
READMITTED = ["<30", ">30", "NO"]
MISSING_TOKEN = "?"  # sentinelle de valeur manquante propre au dataset


RAW_SCHEMA = pa.DataFrameSchema(
    {
        # Identifiants : encounter_id unique, patient_nbr non nul (clé anti-fuite).
        "encounter_id": pa.Column(int, unique=True, nullable=False),
        "patient_nbr": pa.Column(int, nullable=False),
        # Démographie
        "gender": pa.Column(str, pa.Check.isin(GENDERS)),
        "age": pa.Column(str, pa.Check.isin(AGE_BUCKETS)),
        "race": pa.Column(str, pa.Check.isin(RACES), nullable=True),
        # Variables de séjour (entiers positifs / bornés)
        "time_in_hospital": pa.Column(int, pa.Check.in_range(1, 14)),
        "num_lab_procedures": pa.Column(int, pa.Check.ge(0)),
        "num_procedures": pa.Column(int, pa.Check.ge(0)),
        "num_medications": pa.Column(int, pa.Check.ge(0)),
        "number_outpatient": pa.Column(int, pa.Check.ge(0)),
        "number_emergency": pa.Column(int, pa.Check.ge(0)),
        "number_inpatient": pa.Column(int, pa.Check.ge(0)),
        "number_diagnoses": pa.Column(int, pa.Check.ge(0)),
        # Indicateurs traitement
        "change": pa.Column(str, pa.Check.isin(["Ch", "No"])),
        "diabetesMed": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        # Cible
        "readmitted": pa.Column(str, pa.Check.isin(READMITTED), nullable=False),
    },
    strict=False,  # tolère les colonnes non déclarées
    coerce=True,   # harmonise les types (robuste aux dtypes string/Arrow de pandas 3)
    name="diabetes_raw",
)


def validate_raw(df):
    """Valide le DataFrame brut. `lazy=True` collecte TOUTES les erreurs d'un coup."""
    return RAW_SCHEMA.validate(df, lazy=True)

"""Tests du schéma de validation des données brutes."""

import pandas as pd
import pytest

try:
    from pandera.errors import SchemaErrors
except ImportError:  # filet de sécurité selon la version de pandera
    SchemaErrors = Exception

from readmission_risk.validation.schema import validate_raw


def _valid_row(**overrides) -> dict:
    row = {
        "encounter_id": 1,
        "patient_nbr": 100,
        "gender": "Male",
        "age": "[50-60)",
        "race": "Caucasian",
        "time_in_hospital": 3,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 5,
        "change": "No",
        "diabetesMed": "Yes",
        "readmitted": "NO",
    }
    row.update(overrides)
    return row


def test_schema_accepte_donnees_valides():
    df = pd.DataFrame([_valid_row(), _valid_row(encounter_id=2, readmitted="<30")])
    validate_raw(df)  # ne doit pas lever


def test_schema_rejette_gender_invalide():
    df = pd.DataFrame([_valid_row(gender="X"), _valid_row(encounter_id=2)])
    with pytest.raises(SchemaErrors):
        validate_raw(df)


def test_schema_rejette_age_hors_buckets():
    df = pd.DataFrame([_valid_row(age="42"), _valid_row(encounter_id=2)])
    with pytest.raises(SchemaErrors):
        validate_raw(df)


def test_schema_rejette_encounter_id_duplique():
    # Deux séjours avec le même encounter_id viole la contrainte d'unicité.
    df = pd.DataFrame([_valid_row(), _valid_row()])
    with pytest.raises(SchemaErrors):
        validate_raw(df)

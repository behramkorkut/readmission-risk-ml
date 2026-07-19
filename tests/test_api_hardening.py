"""Tests des durcissements post-audit (validation métier /predict, warning auth,
intégrité du modèle, erreurs structurées, manifeste étendu aux modèles).

Complément de tests/test_api_http.py (contrat HTTP de base) : ici on verrouille
les garde-fous ajoutés après la vérification indépendante — sans modifier aucun
test existant.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from readmission_risk.common.config import settings
from readmission_risk.data import manifest as manifest_mod
from readmission_risk.data.manifest import find_model_entry, record_model_artifact, verify
from readmission_risk.serving import api, security

from .helpers import build_demo_model_bundle

VALID_PAYLOAD = {"features": {"num1": 1.5, "cat1": "A"}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient avec un modèle jouet chargé (même pattern que test_api_http)."""
    build_demo_model_bundle(tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "model_filename", "model.joblib")
    monkeypatch.setattr(settings, "data_dir", tmp_path)  # journal de monitoring en zone temporaire
    api._STATE.clear()
    security.reset_rate_limiter()
    with TestClient(api.app) as c:
        yield c
    api._STATE.clear()
    security.reset_rate_limiter()


class TestClinicalValidation:
    """F1 : les champs cliniques connus, quand présents, sont bornés au domaine UCI."""

    @pytest.mark.parametrize(
        "features",
        [
            {"time_in_hospital": -999},  # aberration négative
            {"time_in_hospital": 0},  # hors domaine (min 1 jour)
            {"time_in_hospital": 15},  # hors domaine (max 14 jours)
            {"num_medications": 10**6},  # aberration massive
            {"number_inpatient": 999},  # au-delà du max observé (21)
            {"number_outpatient": -1},  # compteur négatif
            {"time_in_hospital": 5.5},  # compteur non entier
            {"time_in_hospital": "beaucoup"},  # type invalide
            {"age": "37"},  # âge brut : tranches attendues
            {"discharge_disposition_id": 99},  # id hors nomenclature (1-28)
        ],
    )
    def test_out_of_domain_value_422(self, client, features):
        r = client.post("/predict", json={"features": features})
        assert r.status_code == 422
        assert isinstance(r.json()["detail"], list)  # contrat d'erreur structuré FastAPI

    def test_nan_rejected_at_schema_level(self):
        # NaN n'est pas sérialisable en JSON : la garde `isfinite` se teste au niveau schéma.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            api.PredictRequest(features={"time_in_hospital": float("nan")})

    def test_valid_clinical_payload_200(self, client):
        # Le payload de la doc (valeurs cliniques plausibles) reste accepté.
        r = client.post(
            "/predict",
            json={
                "features": {
                    "age": "[70-80)",
                    "time_in_hospital": 5,
                    "number_inpatient": 3,
                    "number_diagnoses": 9,
                    "num_medications": 18,
                    "insulin": "Up",
                    "diabetesMed": "Yes",
                    "discharge_disposition_id": 1,
                }
            },
        )
        assert r.status_code == 200
        assert 0.0 <= r.json()["risk"] <= 1.0

    def test_unknown_keys_tolerated_200(self, client):
        # Choix documenté (api.py) : clés inconnues tolérées, champs connus validés.
        # extra="forbid" impossible : le contrat existant (test_api_http.py) l'exige.
        payload = {"features": {"num1": 1.5, "feature_inconnu": 42}, "clé_en_trop": True}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_none_means_missing_not_error(self, client):
        # None = valeur manquante -> imputée par le pipeline (contrat inchangé).
        r = client.post("/predict", json={"features": {"time_in_hospital": None}})
        assert r.status_code == 200


class TestStructuredErrors:
    """F3 : pannes de chargement / d'inférence -> JSON structuré, jamais de stacktrace."""

    def test_model_load_failure_503_json(self, client, monkeypatch):
        def boom(_path):
            raise RuntimeError("joblib corrompu")

        monkeypatch.setattr(api.joblib, "load", boom)
        api._STATE.clear()  # force un rechargement
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 503
        assert r.headers["content-type"].startswith("application/json")
        assert "Modèle indisponible" in r.json()["detail"]

    def test_inference_failure_500_json(self, client, monkeypatch):
        monkeypatch.setattr(api, "_load_state", lambda: {})  # état incomplet
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 500
        assert r.headers["content-type"].startswith("application/json")
        assert "Erreur d'inférence" in r.json()["detail"]


class TestStartupWarnings:
    """F2/F3 : les fail-open sont assumés mais jamais silencieux."""

    def test_auth_disabled_warning_when_no_api_key(self, monkeypatch):
        from structlog.testing import capture_logs

        monkeypatch.setattr(settings, "api_key", None)
        with capture_logs() as logs:
            api._warn_if_auth_disabled()
        assert any(log["event"] == "auth.disabled_demo" for log in logs)

    def test_no_auth_warning_when_api_key_set(self, monkeypatch):
        from structlog.testing import capture_logs

        monkeypatch.setattr(settings, "api_key", "cle-configuree")
        with capture_logs() as logs:
            api._warn_if_auth_disabled()
        assert not any(log["event"] == "auth.disabled_demo" for log in logs)

    def test_model_integrity_mismatch_warns(self, tmp_path, monkeypatch):
        from structlog.testing import capture_logs

        model = tmp_path / "model.joblib"
        model.write_bytes(b"contenu-quelconque")
        monkeypatch.setattr(api, "find_model_entry", lambda _p: {"sha256": "0" * 64})
        with capture_logs() as logs:
            api._check_model_integrity(model)
        assert any(log["event"] == "model.integrity_mismatch" for log in logs)


class TestModelManifest:
    """F3 : le manifeste SHA-256 couvre aussi les artefacts modèle (hors data_dir)."""

    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        """Mini-racine projet : manifeste dans data/, modèle dans models/."""
        (tmp_path / "data").mkdir()
        (tmp_path / "models").mkdir()
        monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", tmp_path)
        return tmp_path

    def test_record_and_verify_model_roundtrip(self, project):
        model = project / "models" / "model.joblib"
        model.write_bytes(b"fake-model-v1")
        entry = record_model_artifact(model, produced_by="readmission-calibrate")
        assert entry["kind"] == "model"
        assert entry["relpath"] == "models/model.joblib"
        assert len(entry["sha256"]) == 64
        [(name, status, detail)] = verify()
        assert (name, status) == ("model.joblib", "OK")
        assert "octets" in detail  # pas de lignes/cols pour un artefact non-Parquet

    def test_verify_detects_model_tampering(self, project):
        model = project / "models" / "model.joblib"
        model.write_bytes(b"fake-model-v1")
        record_model_artifact(model, produced_by="readmission-calibrate")
        model.write_bytes(b"modele-modifie-hors-chaine")
        assert verify()[0][1] == "MODIFIÉ"

    def test_find_model_entry_matches_by_path_not_name(self, project):
        # Un modèle jouet homonyme hors de l'arborescence enregistrée ne matche pas :
        # pas de faux positif d'intégrité dans les tests de l'API.
        model = project / "models" / "model.joblib"
        model.write_bytes(b"fake-model-v1")
        record_model_artifact(model, produced_by="readmission-calibrate")
        assert find_model_entry(model) is not None
        assert find_model_entry(project / "ailleurs" / "model.joblib") is None

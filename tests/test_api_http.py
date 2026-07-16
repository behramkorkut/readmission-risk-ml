"""Tests d'intégration HTTP de l'API (TestClient FastAPI) — recommandation d'audit n°1.

Complément de tests/test_api.py (appels directs des fonctions = logique métier) :
on passe ici par la vraie couche HTTP — sérialisation/désérialisation JSON, codes
de statut, contrat d'erreur (422 validation, 405 méthode, 503 modèle absent) —
sur le même mini-modèle jouet entraîné à la volée (tests/helpers.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from readmission_risk.common.config import settings
from readmission_risk.serving import api

from .helpers import build_demo_model_bundle

VALID_PAYLOAD = {"features": {"num1": 1.5, "cat1": "A"}}
RESPONSE_KEYS = {"risk", "risk_label", "prediction_set", "confidence_level", "calibration_method", "top_reasons"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient avec un modèle jouet chargé (état de l'API isolé par test)."""
    build_demo_model_bundle(tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "model_filename", "model.joblib")
    api._STATE.clear()
    with TestClient(api.app) as c:
        yield c
    api._STATE.clear()


@pytest.fixture
def client_no_model(tmp_path, monkeypatch):
    """TestClient SANS artefact modèle (models_dir vide -> prédiction impossible)."""
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "model_filename", "model.joblib")
    api._STATE.clear()
    with TestClient(api.app) as c:
        yield c
    api._STATE.clear()


class TestHealthHTTP:
    def test_health_returns_200_json(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert r.json() == {"status": "ok", "model_loaded": True}

    def test_health_reflects_missing_model(self, client_no_model):
        r = client_no_model.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "model_loaded": False}


class TestPredictHTTP:
    def test_predict_valid_payload_200(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        # Contrat de sortie complet (PredictResponse sérialisé en JSON).
        assert set(body) == RESPONSE_KEYS
        assert 0.0 <= body["risk"] <= 1.0
        assert body["risk_label"] in {"non_readmis", "readmission_30j"}
        assert body["prediction_set"]  # ensemble conformel non vide
        assert set(body["prediction_set"]) <= {"non_readmis", "readmission_30j"}
        assert body["confidence_level"] == pytest.approx(0.9)
        assert body["calibration_method"] in {"isotonic", "sigmoid"}
        assert 1 <= len(body["top_reasons"]) <= 5  # top-5 SHAP (borné par le nb de features)
        reason = body["top_reasons"][0]
        assert set(reason) == {"feature", "contribution", "direction"}
        assert reason["direction"] in {"augmente", "diminue"}

    def test_predict_unknown_features_and_extra_keys_tolerated(self, client):
        # Features inconnues : ignorées (le pipeline n'utilise que ses colonnes).
        # Clé de premier niveau en trop : ignorée (Pydantic, extra='ignore').
        payload = {"features": {"num1": 1.5, "feature_inconnu": 42}, "clé_en_trop": True}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert 0.0 <= r.json()["risk"] <= 1.0

    def test_predict_empty_features_200(self, client):
        # Toutes les features manquantes -> imputées par le pipeline (contrat API).
        r = client.post("/predict", json={"features": {}})
        assert r.status_code == 200

    def test_predict_missing_features_field_422(self, client):
        r = client.post("/predict", json={})  # 'features' est requis
        assert r.status_code == 422
        assert isinstance(r.json()["detail"], list)  # contrat d'erreur FastAPI

    def test_predict_wrong_features_type_422(self, client):
        r = client.post("/predict", json={"features": ["pas", "un", "dict"]})
        assert r.status_code == 422

    def test_predict_malformed_json_422(self, client):
        r = client.post("/predict", content=b"{json casse", headers={"content-type": "application/json"})
        assert r.status_code == 422

    def test_predict_wrong_method_405(self, client):
        r = client.get("/predict")  # POST uniquement
        assert r.status_code == 405


class TestPredictModelMissing:
    def test_predict_503_when_model_absent(self, client_no_model):
        r = client_no_model.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 503
        assert "Modèle absent" in r.json()["detail"]


class TestOpenAPI:
    def test_openapi_schema_served(self, client):
        r = client.get("/openapi.json")  # doc Swagger auto-générée accessible
        assert r.status_code == 200
        assert "/predict" in r.json()["paths"]
        assert "/health" in r.json()["paths"]
        assert "/ready" in r.json()["paths"]


class TestReadyHTTP:
    """Readiness avancée (audit n°10) : chargement réel + prédiction factice + mémoire."""

    def test_ready_200_with_smoke_test_and_memory(self, client):
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert body["status"] == "ready"
        assert body["model_loaded"] is True
        assert body["smoke_test_ok"] is True
        assert 0.0 <= body["smoke_test_risk"] <= 1.0  # prédiction factice = proba valide
        assert body["memory_mb_max_rss"] > 0.0

    def test_ready_503_when_model_absent(self, client_no_model):
        r = client_no_model.get("/ready")
        assert r.status_code == 503
        assert "non prêt" in r.json()["detail"]

    def test_ready_is_public_under_api_key(self, tmp_path, monkeypatch):
        # Les sondes d'orchestration ne portent pas de clé : /ready reste public,
        # comme /health, même quand l'auth est activée sur /predict.
        build_demo_model_bundle(tmp_path / "model.joblib")
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        monkeypatch.setattr(settings, "model_filename", "model.joblib")
        monkeypatch.setattr(settings, "api_key", "cle-quelconque")
        api._STATE.clear()
        with TestClient(api.app) as c:
            assert c.get("/ready").status_code == 200
            assert c.post("/predict", json=VALID_PAYLOAD).status_code == 401
        api._STATE.clear()

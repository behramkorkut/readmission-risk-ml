"""Tests de la sécurité HTTP de l'API : clé API (401/403) + rate limiting (429).

Comportement nominal documenté (cf. serving/security.py) : tant que
`settings.api_key` est None, /predict reste ouvert (démo publique) ; dès qu'une
clé est configurée, l'en-tête X-API-Key est exigé. Le rate limiting est une
fenêtre glissante en mémoire par IP client, testée ici sans `sleep` grâce à
l'horloge injectable du limiteur.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from readmission_risk.common.config import settings
from readmission_risk.serving import api, security
from readmission_risk.serving.security import SlidingWindowRateLimiter

from .helpers import build_demo_model_bundle

PAYLOAD = {"features": {"num1": 1.5, "cat1": "A"}}
TEST_KEY = "cle-secrete-de-test"


@pytest.fixture
def secured_client(tmp_path, monkeypatch):
    """TestClient : modèle jouet chargé + clé API configurée + limiteur vierge."""
    build_demo_model_bundle(tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "model_filename", "model.joblib")
    monkeypatch.setattr(settings, "data_dir", tmp_path)  # journal de monitoring en zone temporaire
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    security.reset_rate_limiter()
    api._STATE.clear()
    with TestClient(api.app) as c:
        yield c
    api._STATE.clear()
    security.reset_rate_limiter()


class TestApiKeyAuth:
    def test_health_stays_public(self, secured_client):
        r = secured_client.get("/health")  # sondes : jamais d'auth
        assert r.status_code == 200

    def test_predict_without_key_401(self, secured_client):
        r = secured_client.post("/predict", json=PAYLOAD)
        assert r.status_code == 401
        assert "X-API-Key" in r.json()["detail"]

    def test_predict_wrong_key_403(self, secured_client):
        r = secured_client.post("/predict", json=PAYLOAD, headers={"X-API-Key": "mauvaise-cle"})
        assert r.status_code == 403

    def test_predict_with_key_200(self, secured_client):
        r = secured_client.post("/predict", json=PAYLOAD, headers={"X-API-Key": TEST_KEY})
        assert r.status_code == 200
        assert 0.0 <= r.json()["risk"] <= 1.0

    def test_predict_open_when_no_key_configured(self, secured_client, monkeypatch):
        # Comportement démo : sans clé configurée, l'endpoint reste ouvert.
        monkeypatch.setattr(settings, "api_key", None)
        r = secured_client.post("/predict", json=PAYLOAD)
        assert r.status_code == 200


class TestRateLimitHTTP:
    def test_429_after_threshold_with_retry_after(self, secured_client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
        headers = {"X-API-Key": TEST_KEY}
        for _ in range(3):
            assert secured_client.post("/predict", json=PAYLOAD, headers=headers).status_code == 200
        r = secured_client.post("/predict", json=PAYLOAD, headers=headers)
        assert r.status_code == 429
        assert "retry-after" in {k.lower(): v for k, v in r.headers.items()}
        assert "limite" in r.json()["detail"]

    def test_rate_limit_before_auth(self, secured_client, monkeypatch):
        # Une rafale SANS clé plafonne aussi (le 429 précède le 401) : brute-force freiné.
        monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
        assert secured_client.post("/predict", json=PAYLOAD).status_code == 401
        assert secured_client.post("/predict", json=PAYLOAD).status_code == 401
        assert secured_client.post("/predict", json=PAYLOAD).status_code == 429

    def test_rate_limit_disabled(self, secured_client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
        headers = {"X-API-Key": TEST_KEY}
        for _ in range(10):
            assert secured_client.post("/predict", json=PAYLOAD, headers=headers).status_code == 200


class TestSlidingWindowUnit:
    """Logique du limiteur, horloge injectée (aucun sleep, aucun réseau)."""

    def test_window_slides_and_clients_are_isolated(self):
        lim = SlidingWindowRateLimiter()
        assert lim.check("ip", 2, 60.0, now=100.0) == 0.0  # 1er hit : accepté
        assert lim.check("ip", 2, 60.0, now=101.0) == 0.0  # 2e hit : accepté
        retry = lim.check("ip", 2, 60.0, now=102.0)  # 3e hit : refusé
        assert retry == pytest.approx(58.0)  # libéré quand le 1er hit sort (t=160)
        assert lim.check("ip", 2, 60.0, now=161.0) == 0.0  # fenêtre glissée : accepté
        assert lim.check("autre-ip", 2, 60.0, now=162.0) == 0.0  # isolation par client

    def test_reset_clears_state(self):
        lim = SlidingWindowRateLimiter()
        lim.check("ip", 1, 60.0, now=100.0)
        assert lim.check("ip", 1, 60.0, now=101.0) > 0.0
        lim.reset()
        assert lim.check("ip", 1, 60.0, now=101.0) == 0.0

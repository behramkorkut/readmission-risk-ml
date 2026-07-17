"""Tests du monitoring online des prédictions (audit n°6).

Deux niveaux :
- unitaires : store SQLite (roundtrip, fenêtre temporelle via horloge injectée,
  état vide, robustesse — la journalisation ne lève jamais), contrat privacy-by-design
  (aucune colonne de feature dans le journal) ;
- HTTP : GET /monitoring/summary (état vide, après prédictions réelles, auth 401,
  validation du paramètre de fenêtre).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from readmission_risk.common.config import settings
from readmission_risk.monitoring.predictions_log import load_summary, log_prediction
from readmission_risk.serving import api

from .helpers import build_demo_model_bundle

TEST_KEY = "cle-monitoring-test"


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Redirige le journal SQLite vers tmp_path (herméticité des tests)."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


@pytest.fixture
def http_client(tmp_path, monkeypatch):
    """TestClient : modèle jouet + journal isolé + clé API configurée."""
    build_demo_model_bundle(tmp_path / "model.joblib")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(settings, "model_filename", "model.joblib")
    monkeypatch.setattr(settings, "data_dir", tmp_path)  # journal SQLite en zone temporaire
    monkeypatch.setattr(settings, "api_key", TEST_KEY)
    api._STATE.clear()
    with TestClient(api.app) as c:
        yield c
    api._STATE.clear()


class TestPredictionLogUnit:
    def test_log_and_summary_roundtrip(self, log_dir):
        log_prediction(latency_ms=12.0, risk=0.10, risk_label="non_readmis", prediction_set_size=1)
        log_prediction(latency_ms=20.0, risk=0.50, risk_label="readmission_30j", prediction_set_size=2)
        log_prediction(latency_ms=28.0, risk=0.90, risk_label="readmission_30j", prediction_set_size=1)
        s = load_summary(window_hours=24.0)
        assert s["n_predictions"] == 3
        assert s["risk"]["mean"] == pytest.approx(0.5)
        assert s["risk"]["p50"] == pytest.approx(0.5)
        assert s["risk"]["p90"] == pytest.approx(0.9)
        assert sum(s["risk"]["histogram"]["counts"]) == 3
        assert s["latency_ms"]["mean"] == pytest.approx(20.0)
        assert s["latency_ms"]["p95"] == pytest.approx(28.0)
        assert s["uncertainty_rate"] == pytest.approx(1 / 3, abs=1e-3)  # 1 ensemble à 2 labels
        assert s["expected_error_bound"] == pytest.approx(1 - settings.conformal_confidence)

    def test_window_filtering(self, log_dir):
        old = datetime.now(UTC) - timedelta(hours=48)
        log_prediction(latency_ms=5.0, risk=0.2, risk_label="non_readmis", prediction_set_size=1, ts=old)
        log_prediction(latency_ms=5.0, risk=0.3, risk_label="non_readmis", prediction_set_size=1)
        assert load_summary(window_hours=24.0)["n_predictions"] == 1  # le vieux point est exclu
        assert load_summary(window_hours=72.0)["n_predictions"] == 2  # fenêtre plus large : inclus

    def test_empty_summary_when_no_data(self, log_dir):
        s = load_summary(window_hours=24.0)
        assert s["n_predictions"] == 0
        assert s["risk"] is None and s["latency_ms"] is None and s["uncertainty_rate"] is None
        assert s["expected_error_bound"] == pytest.approx(0.1)  # borne conformelle toujours exposée

    def test_logging_never_raises(self, monkeypatch):
        # Chemin impossible : la journalisation avale l'erreur (jamais de 500 pour du monitoring).
        monkeypatch.setattr(settings, "data_dir", Path("/proc/chemin-impossible"))
        log_prediction(latency_ms=1.0, risk=0.5, risk_label="non_readmis", prediction_set_size=1)

    def test_privacy_no_features_logged(self, log_dir):
        # Contrat privacy-by-design : le journal ne contient QUE des sorties + métadonnées.
        log_prediction(latency_ms=1.0, risk=0.5, risk_label="non_readmis", prediction_set_size=1)
        db = log_dir / settings.predictions_log_filename
        with sqlite3.connect(db) as con:
            cols = {row[1] for row in con.execute("PRAGMA table_info(predictions)")}
        assert cols == {"id", "ts_utc", "latency_ms", "risk", "risk_label", "prediction_set_size"}


class TestMonitoringSummaryHTTP:
    def test_summary_empty_200(self, http_client):
        r = http_client.get("/monitoring/summary", headers={"X-API-Key": TEST_KEY})
        assert r.status_code == 200
        body = r.json()
        assert body["n_predictions"] == 0
        assert body["risk"] is None
        assert body["window_hours"] == pytest.approx(24.0)

    def test_summary_after_real_predictions(self, http_client):
        headers = {"X-API-Key": TEST_KEY}
        for _ in range(2):
            assert http_client.post("/predict", json={"features": {"num1": 1.0}}, headers=headers).status_code == 200
        r = http_client.get("/monitoring/summary", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["n_predictions"] == 2
        assert 0.0 <= body["risk"]["mean"] <= 1.0
        assert body["latency_ms"]["mean"] > 0.0  # latence réellement mesurée
        assert sum(body["risk"]["histogram"]["counts"]) == 2
        assert 0.0 <= body["uncertainty_rate"] <= 1.0

    def test_summary_requires_api_key(self, http_client):
        assert http_client.get("/monitoring/summary").status_code == 401

    def test_summary_window_validation(self, http_client):
        r = http_client.get("/monitoring/summary?window_hours=0", headers={"X-API-Key": TEST_KEY})
        assert r.status_code == 422  # fenêtre strictement positive (Query gt=0)

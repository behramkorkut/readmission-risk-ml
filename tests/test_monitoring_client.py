"""Tests du client de monitoring (urllib mocké — aucun réseau, aucune dépendance UI).

Vérifie le contrat de fetch_monitoring_summary : succès, gestion gracieuse des
erreurs (HTTP 401, connexion refusée, JSON invalide, contrat inattendu) et passage
de la clé API dans l'en-tête.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from readmission_risk.monitoring.client import fetch_monitoring_summary

PAYLOAD = {
    "window_hours": 24.0,
    "generated_at_utc": "2026-07-16T12:00:00+00:00",
    "n_predictions": 12,
    "risk": {"mean": 0.13, "p50": 0.09, "p90": 0.31, "histogram": {"bin_edges": [0.0, 1.0], "counts": [12]}},
    "latency_ms": {"mean": 8.5, "p50": 7.0, "p95": 15.0},
    "uncertainty_rate": 0.08,
    "expected_error_bound": 0.1,
}


class _FakeResponse:
    """Réponse HTTP factice (context manager + read(), comme urlopen)."""

    def __init__(self, payload: dict):
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body.read()


def _patch_urlopen(monkeypatch, behavior):
    """Remplace urlopen par `behavior(req)` (retourne une réponse ou lève)."""
    calls: list[urllib.request.Request] = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return behavior(req)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_fetch_success_returns_payload(monkeypatch):
    _patch_urlopen(monkeypatch, lambda req: _FakeResponse(PAYLOAD))
    out = fetch_monitoring_summary("https://api.example.com/", api_key="k", window_hours=24.0)
    assert out == PAYLOAD  # '/' final toléré dans l'URL de base


def test_fetch_sends_api_key_header(monkeypatch):
    calls = _patch_urlopen(monkeypatch, lambda req: _FakeResponse(PAYLOAD))
    fetch_monitoring_summary("https://api.example.com", api_key="ma-cle")
    assert calls[0].get_header("X-api-key") == "ma-cle"  # urllib normalise la casse


def test_fetch_without_key_sends_no_header(monkeypatch):
    calls = _patch_urlopen(monkeypatch, lambda req: _FakeResponse(PAYLOAD))
    fetch_monitoring_summary("https://api.example.com", api_key=None)
    assert calls[0].get_header("X-api-key") is None


def test_fetch_401_returns_none(monkeypatch):
    def raise_401(req):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    _patch_urlopen(monkeypatch, raise_401)
    assert fetch_monitoring_summary("https://api.example.com", api_key=None) is None


def test_fetch_connection_error_returns_none(monkeypatch):
    def raise_conn(req):
        raise urllib.error.URLError("connexion refusée")

    _patch_urlopen(monkeypatch, raise_conn)
    assert fetch_monitoring_summary("https://api.example.com", api_key=None) is None


def test_fetch_unexpected_payload_returns_none(monkeypatch):
    _patch_urlopen(monkeypatch, lambda req: _FakeResponse({"hello": "world"}))
    assert fetch_monitoring_summary("https://api.example.com", api_key=None) is None

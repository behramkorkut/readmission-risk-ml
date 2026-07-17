"""Client léger pour `GET /monitoring/summary` de l'API (consommé par l'UI Streamlit).

Stdlib uniquement (urllib) : aucune dépendance ajoutée — l'app Streamlit Cloud
s'installe depuis requirements.txt et la CI ne connaît que les deps du projet.
Toute erreur (réseau, HTTP 4xx/5xx, JSON invalide, contrat inattendu) -> None :
l'appelant affiche alors un état dégradé gracieux plutôt qu'une stacktrace.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

EXPECTED_KEYS = {"n_predictions", "risk", "latency_ms", "uncertainty_rate", "expected_error_bound"}


def fetch_monitoring_summary(
    base_url: str,
    api_key: str | None,
    window_hours: float = 24.0,
    timeout: float = 5.0,
) -> dict | None:
    """Récupère le résumé de monitoring de l'API. Renvoie le dict JSON, ou None.

    `api_key` est optionnel : si l'API exige une clé et qu'aucune n'est fournie,
    l'appel échoue proprement (None) — c'est l'état « non configuré » de la démo.
    """
    url = f"{base_url.rstrip('/')}/monitoring/summary?window_hours={window_hours}"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    # Contrat minimal : les clés attendues sont présentes (sinon, ce n'est pas notre API).
    return payload if isinstance(payload, dict) and EXPECTED_KEYS <= set(payload) else None

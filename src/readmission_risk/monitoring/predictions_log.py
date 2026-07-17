"""Journal et agrégation des prédictions servies (monitoring online, audit n°6).

Complément du drift Evidently (offline, features) : ici on suit le COMPORTEMENT du
modèle en production — volume, distribution des scores, latences, incertitude.

Privacy-by-design : on ne journalise QUE les sorties (risque, label, taille de
l'ensemble conformel, latence, horodatage). JAMAIS les features d'entrée — en
santé, la minimisation des données (esprit RGPD) prime : détecter une dérive du
comportement du modèle n'exige pas de savoir quel patient était concerné.

Stockage : SQLite dédié (cohérent avec le backend MLflow du projet), une connexion
par opération (sûr en threadpool FastAPI). Règle d'or : la journalisation ne doit
JAMAIS faire échouer une prédiction — toute erreur est avalée avec un warning.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog

from readmission_risk.common.config import settings

log = structlog.get_logger()

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,               -- ISO 8601 UTC (comparaison lexicographique)
    latency_ms REAL NOT NULL,
    risk REAL NOT NULL,
    risk_label TEXT NOT NULL,
    prediction_set_size INTEGER NOT NULL
)
"""


def _db_path() -> Path:
    return settings.data_dir / settings.predictions_log_filename


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(_CREATE_TABLE)
    return con


def log_prediction(
    *,
    latency_ms: float,
    risk: float,
    risk_label: str,
    prediction_set_size: int,
    ts: datetime | None = None,  # horloge injectable (tests sans sleep)
) -> None:
    """Journalise une prédiction servie. Ne lève JAMAIS d'exception."""
    try:
        ts = ts or datetime.now(UTC)
        with _connect() as con:
            con.execute(
                "INSERT INTO predictions (ts_utc, latency_ms, risk, risk_label, prediction_set_size)"
                " VALUES (?, ?, ?, ?, ?)",
                (ts.isoformat(), latency_ms, risk, risk_label, prediction_set_size),
            )
    except Exception:
        log.warning("predictions_log.write_failed", exc_info=True)


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Percentile par rang le plus proche (q ∈ [0, 1]), liste triée non vide."""
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _histogram(risks: list[float], n_bins: int = 10) -> dict:
    """Histogramme des risques sur [0, 1] (dernier bin inclusif de 1.0)."""
    edges = [round(i / n_bins, 1) for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for r in risks:
        counts[min(int(r * n_bins), n_bins - 1)] += 1
    return {"bin_edges": edges, "counts": counts}


def load_summary(window_hours: float = 24.0) -> dict:
    """Agrège les prédictions journalisées sur la fenêtre glissante demandée.

    Sans donnée (fenêtre vide ou journal absent), les blocs de stats sont None :
    un « pas de données » explicite vaut mieux que des zéros trompeurs.
    """
    cutoff = (datetime.now(UTC) - timedelta(hours=window_hours)).isoformat()
    rows: list[tuple[float, float, int]] = []
    if _db_path().exists():
        try:
            with _connect() as con:
                rows = con.execute(
                    "SELECT risk, latency_ms, prediction_set_size FROM predictions WHERE ts_utc >= ?",
                    (cutoff,),
                ).fetchall()
        except Exception:
            log.warning("predictions_log.read_failed", exc_info=True)

    n = len(rows)
    summary: dict = {
        "window_hours": window_hours,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_predictions": n,
        "risk": None,
        "latency_ms": None,
        "uncertainty_rate": None,
        # Borne d'erreur attendue, statistiquement fondée : garantie de couverture conformelle.
        "expected_error_bound": round(1 - settings.conformal_confidence, 4),
    }
    if n == 0:
        return summary

    risks = sorted(r for r, _, _ in rows)
    latencies = sorted(lat for _, lat, _ in rows)
    uncertain = sum(1 for _, _, size in rows if size == 2)  # ensemble {non, oui} = hésitation
    summary["risk"] = {
        "mean": round(sum(risks) / n, 4),
        "p50": round(_percentile(risks, 0.5), 4),
        "p90": round(_percentile(risks, 0.9), 4),
        "histogram": _histogram(risks),
    }
    summary["latency_ms"] = {
        "mean": round(sum(latencies) / n, 2),
        "p50": round(_percentile(latencies, 0.5), 2),
        "p95": round(_percentile(latencies, 0.95), 2),
    }
    summary["uncertainty_rate"] = round(uncertain / n, 4)
    return summary

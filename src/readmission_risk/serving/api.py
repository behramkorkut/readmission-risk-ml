"""API de scoring (FastAPI) : sert le modèle calibré + conformal + raisons SHAP.

Endpoints :
- GET  /health   : vivacité (liveness) — léger, le processus tourne ; pour sondes fréquentes
- GET  /ready    : disponibilité (readiness) — modèle chargé + prédiction factice + mémoire
- POST /predict  : risque calibré + ensemble de prédiction conformel + top raisons SHAP
- GET  /monitoring/summary : monitoring online des prédictions servies (audit n°6)

Le modèle (models/model.joblib) est produit par `readmission-calibrate`. Il contient
le modèle calibré, le prédicteur conformel et le pipeline de base (pour SHAP).
"""

from __future__ import annotations

import resource
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from pydantic import BaseModel, Field

from readmission_risk.common.config import settings
from readmission_risk.monitoring.predictions_log import load_summary, log_prediction
from readmission_risk.serving.security import rate_limit, require_api_key

app = FastAPI(title="Readmission Risk API", version="1.0")

# État chargé une seule fois (modèle + explainer), à la première requête.
_STATE: dict[str, Any] = {}

LABELS = {0: "non_readmis", 1: "readmission_30j"}


def _load_state() -> dict[str, Any]:
    if not _STATE:
        path = settings.models_dir / settings.model_filename
        if not path.exists():
            raise FileNotFoundError(f"Modèle absent : {path}. Lance d'abord readmission-calibrate.")
        bundle = joblib.load(path)
        base = bundle["base_pipeline"]
        _STATE.update(bundle)
        _STATE["prep"] = base.named_steps["prep"]
        _STATE["explainer"] = shap.TreeExplainer(base.named_steps["clf"])
        _STATE["feat_names"] = list(_STATE["prep"].get_feature_names_out())
    return _STATE


# ---------- Schémas ----------
class PredictRequest(BaseModel):
    # On accepte un dictionnaire feature -> valeur (colonnes manquantes -> imputées).
    features: dict[str, Any] = Field(
        ...,
        examples=[{"age": "[70-80)", "time_in_hospital": 5, "number_inpatient": 3,
                   "number_diagnoses": 9, "num_medications": 18, "insulin": "Up",
                   "diabetesMed": "Yes", "discharge_disposition_id": 1}],
    )


class Reason(BaseModel):
    feature: str
    contribution: float
    direction: str  # "augmente" / "diminue" le risque


class PredictResponse(BaseModel):
    risk: float
    risk_label: str
    prediction_set: list[str]
    confidence_level: float
    calibration_method: str
    top_reasons: list[Reason]


def _max_rss_mb() -> float:
    """Mémoire résidente max du processus (Mo) — info de readiness, pas un garde-fou."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux : ru_maxrss en Kio ; macOS : en octets.
    return round(usage / (1024**2 if sys.platform == "darwin" else 1024), 1)


# ---------- Endpoints ----------
@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness : léger et toujours répondant (le processus tourne, le fichier existe)."""
    path = settings.models_dir / settings.model_filename
    return {"status": "ok", "model_loaded": path.exists()}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness : le modèle est RÉELLEMENT chargé et capable de prédire.

    Charge l'état si besoin (modèle + explainer SHAP) puis exécute une prédiction
    factice (toutes les features absentes -> imputées, contrat de /predict).
    Toute défaillance -> 503, pour qu'un orchestrateur retire le service du trafic.
    """
    try:
        state = _load_state()
    except Exception as exc:  # fichier absent, joblib corrompu, explainer SHAP en échec
        raise HTTPException(status_code=503, detail=f"Service non prêt : {exc}") from exc
    try:
        row = {col: np.nan for col in state["feature_cols"]}
        df = pd.DataFrame([row], columns=state["feature_cols"])
        proba = float(state["model"].predict_proba(df)[:, 1][0])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Modèle chargé mais prédiction impossible : {exc}") from exc
    return {
        "status": "ready",
        "model_loaded": True,
        "smoke_test_ok": 0.0 <= proba <= 1.0,
        "smoke_test_risk": round(proba, 4),
        "memory_mb_max_rss": _max_rss_mb(),
    }


# /predict est PUBLIC (démo portfolio) : sa protection est le rate limiting par IP.
# Sur des données de santé réelles, on ajouterait Security(require_api_key) ici
# (une ligne — la capacité d'auth existe et est testée, cf. /monitoring/summary).
@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(rate_limit)],
)
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()  # latence de service (monitoring online)
    try:
        state = _load_state()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 1 ligne avec toutes les colonnes attendues (absentes -> NaN, imputées par le pipeline).
    row = {col: req.features.get(col, np.nan) for col in state["feature_cols"]}
    df = pd.DataFrame([row], columns=state["feature_cols"])

    # Probabilité calibrée
    proba = float(state["model"].predict_proba(df)[:, 1][0])

    # Ensemble de prédiction conformel
    _, y_set = state["conformal"].predict_set(df)
    members = y_set[0, :, 0] if y_set.ndim == 3 else y_set[0]
    prediction_set = [LABELS[i] for i, inside in enumerate(members) if inside]

    # Raisons SHAP (sur le pipeline de base)
    x = state["prep"].transform(df)
    x_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    sv = state["explainer"].shap_values(x_dense)
    if isinstance(sv, list):
        sv = sv[1]
    contrib = np.asarray(sv)[0]
    top_idx = np.argsort(np.abs(contrib))[-5:][::-1]
    reasons = [
        Reason(
            feature=state["feat_names"][i],
            contribution=round(float(contrib[i]), 4),
            direction="augmente" if contrib[i] > 0 else "diminue",
        )
        for i in top_idx
    ]

    resp = PredictResponse(
        risk=round(proba, 4),
        risk_label=LABELS[int(proba >= 0.5)],
        prediction_set=prediction_set,
        confidence_level=float(state["confidence_level"]),
        calibration_method=str(state["calibration_method"]),
        top_reasons=reasons,
    )
    # Monitoring online : sorties + latence uniquement (jamais les features). Ne lève jamais.
    log_prediction(
        latency_ms=(time.perf_counter() - t0) * 1000,
        risk=resp.risk,
        risk_label=resp.risk_label,
        prediction_set_size=len(resp.prediction_set),
    )
    return resp


@app.get("/monitoring/summary", dependencies=[Security(require_api_key)])
def monitoring_summary(window_hours: float = Query(24.0, gt=0, le=24 * 30)) -> dict[str, Any]:
    """Monitoring online des prédictions servies : volume, distribution du risque,
    latences, incertitude conformelle — sur une fenêtre glissante (défaut 24 h).
    Agrégats anonymisés : aucune donnée patient n'est journalisée (ni exposée)."""
    return load_summary(window_hours)


def run() -> None:
    """Point d'entrée `readmission-serve` : lance le serveur uvicorn."""
    import uvicorn

    uvicorn.run("readmission_risk.serving.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()

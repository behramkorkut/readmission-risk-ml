# Readmission Risk — Prédiction de réadmission hospitalière à 30 jours

Projet **Machine Learning / Data Science** de bout en bout : prédire le risque de
réadmission à moins de 30 jours pour des patients diabétiques, à partir de données
hospitalières **réelles et imparfaites**. L'accent est mis sur la **rigueur** (validation
anti-fuite), la **confiance** (probabilités calibrées + incertitude conformelle),
l'**explicabilité** (SHAP), l'**équité**, et un socle **MLOps** complet (suivi
d'expériences, API, monitoring de drift, CI, Docker).

## Enjeu

Les réadmissions précoces sont coûteuses, souvent évitables et pénalisées financièrement.
Un modèle utile doit être **honnête** (pas de fuite de données) et **digne de confiance**
(calibré, explicable, auditable, surveillé en production).

## Résultats clés (test hold-out, patient-disjoint)

| Métrique | Baseline (régression log.) | LightGBM tuné + calibré |
|----------|----------------------------|--------------------------|
| **PR-AUC** (primaire ; base 0,114) | 0,215 | **0,231** |
| **ROC-AUC** | 0,664 | **0,675** |
| **Brier** (calibré) | 0,225 | **0,097** |
| **Couverture conformelle** (cible 90 %) | — | **~0,90** |

Variable la plus prédictive (SHAP) : `number_inpatient` (hospitalisations antérieures).
Audit d'équité : équitable selon le sexe ; dégradation documentée sur les patients très
âgés. Détails et limites dans la [**Model Card**](MODEL_CARD.md).

## Données

*Diabetes 130-US hospitals (1999-2008)* — ~101 766 séjours, 50 variables (UCI, accès libre).
Choisi pour son **réalisme** : valeurs manquantes, codes ICD-9, plusieurs séjours par
patient (risque de fuite), fuites de cible (décès/soins palliatifs) à neutraliser.

## Stack technique

| Domaine             | Outils                                              |
|---------------------|-----------------------------------------------------|
| Données / features  | pandas, numpy, scikit-learn, pyarrow                |
| Validation données  | Pandera                                             |
| Modélisation        | LightGBM, Optuna                                    |
| Confiance           | calibration (sklearn), conformal prediction (MAPIE) |
| Explicabilité/équité| SHAP, audit par sous-groupes                        |
| MLOps               | MLflow, Evidently (drift), Docker, CI GitHub Actions|
| Service             | FastAPI                                             |
| Gestion de projet   | uv                                                  |

## Installation

```bash
uv sync --group dev       # environnement + outils de dev
cp .env.example .env
```

## Pipeline complet (de la donnée au service)

```bash
uv run readmission-ingest        # 1. télécharge le dataset (UCI) -> data/
uv run readmission-clean         # 2. nettoyage + anti-fuite -> data/diabetic_clean.parquet
uv run readmission-train-baseline# 3. baseline (régression logistique) + MLflow
uv run readmission-train-gboost  # 4. LightGBM tuné par Optuna + MLflow
uv run readmission-calibrate     # 5. calibration + conformal -> models/model.joblib
uv run readmission-explain       # 6. SHAP + audit d'équité -> reports/
uv run readmission-drift         # 7. monitoring de dérive -> reports/drift_report.html
uv run readmission-serve         # 8. API de scoring -> http://localhost:8000/docs
```

Suivi des expériences : `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`

### Exemple d'appel à l'API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features":{"age":"[70-80)","number_inpatient":3,"number_diagnoses":9}}'
# -> {"risk":..., "prediction_set":[...], "confidence_level":0.9, "top_reasons":[...]}
```

## Docker (API)

```bash
uv run readmission-calibrate          # génère models/model.joblib (embarqué dans l'image)
docker build -t readmission-api .
docker run -p 8000:8000 readmission-api
```

## Qualité

```bash
uv run ruff check .       # lint
uv run pytest -q          # tests (validés aussi en CI à chaque push)
```

## Structure

```
readmission-risk-ml/
├── src/readmission_risk/
│   ├── common/      # config (graine, chemins, MLflow)
│   ├── data/        # ingestion, nettoyage/anti-fuite, split par patient
│   ├── validation/  # schéma Pandera (contrat de données)
│   ├── features/    # ColumnTransformer (imputation, OHE, anti-fuite intra-CV)
│   ├── modeling/    # CV groupée, baseline, LightGBM+Optuna, calibration+conformal
│   ├── evaluation/  # SHAP + audit d'équité
│   ├── serving/     # API FastAPI
│   └── monitoring/  # drift (Evidently) + injection de dérive
├── tests/           # tests pytest (anti-fuite, schéma, API, drift...)
├── reports/         # graphiques (calibration, SHAP) + rapport de drift
├── MODEL_CARD.md    # usage, performances, équité, limites
└── journal/         # journal de bord détaillé (démarche pas à pas)
```

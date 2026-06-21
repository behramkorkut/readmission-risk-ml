# Readmission Risk — Prédiction de réadmission hospitalière à 30 jours

Projet **Machine Learning / Data Science** : prédire le risque de réadmission à
moins de 30 jours pour des patients diabétiques, à partir de données hospitalières
réelles et imparfaites. L'accent est mis sur une **validation rigoureuse** (anti-fuite
de données), des **prédictions calibrées et accompagnées d'une incertitude**
(conformal prediction), l'**explicabilité** (SHAP), l'**équité**, et un socle **MLOps**.

## Enjeu

Les réadmissions précoces sont coûteuses, souvent évitables et pénalisées
financièrement. Un modèle fiable permet de cibler les interventions post-sortie
sur les patients à risque — à condition d'être **honnête** (pas de fuite de données)
et **digne de confiance** (calibré, explicable, auditable).

## Données

*Diabetes 130-US hospitals (1999-2008)* — ~101 766 séjours, 50 variables
(UCI Machine Learning Repository, accès libre). Jeu de données volontairement
choisi pour son réalisme : valeurs manquantes, codes diagnostics ICD-9, plusieurs
séjours par patient (risque de fuite), et fuites de cible à neutraliser.

## Stack technique (au fil des étapes)

| Domaine            | Outils                                                        |
|--------------------|---------------------------------------------------------------|
| Données / features | pandas, numpy, scikit-learn, pyarrow                          |
| Validation données | Pandera                                                       |
| Modélisation       | scikit-learn, LightGBM/XGBoost, Optuna                        |
| Confiance          | calibration (sklearn), conformal prediction (MAPIE)          |
| Explicabilité/équité| SHAP, audit par sous-groupes                                 |
| MLOps              | MLflow, Evidently (drift), Docker, CI GitHub Actions          |
| Service            | FastAPI                                                       |
| Gestion de projet  | uv                                                            |

## Démarrage

```bash
uv sync                  # installe l'environnement
uv sync --group dev      # outils de dev (ruff, pytest)
cp .env.example .env      # config locale
```

## Structure

```
readmission-risk-ml/
├── src/readmission_risk/
│   ├── common/      # config, utilitaires
│   ├── data/        # ingestion / chargement
│   ├── validation/  # schéma Pandera, contrôles qualité
│   ├── features/    # pipeline de préparation
│   ├── modeling/    # entraînement, tuning
│   ├── evaluation/  # métriques, calibration, conformal, SHAP, équité
│   ├── serving/     # API FastAPI
│   └── monitoring/  # drift (Evidently), injection de défauts
├── data/            # données locales (non versionnées)
├── models/          # modèles entraînés (non versionnés)
├── reports/         # rapports générés (non versionnés)
├── tests/           # tests pytest
└── pyproject.toml
```

Le projet est construit pas à pas — voir le journal de bord.

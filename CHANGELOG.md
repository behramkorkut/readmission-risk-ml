# Changelog

Toutes les évolutions notables du projet sont consignées dans ce fichier.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le
versionnage est [sémantique](https://semver.org/lang/fr/) (MAJEUR.MINEUR.CORRECTIF).

## [1.1.0] - 2026-07-17

Chantier post-audit externe (rapport d'audit du 2026-07-16, note 18,5/20) :
implémentation des recommandations n°1, 2, 3, 4, 6 et 10, en phases commitées
séparément. Suite de tests : **33 → 85**.

### Ajouté

- **Tests d'intégration HTTP** de l'API (TestClient FastAPI) : contrat JSON,
  codes 200/422/405/503, doc OpenAPI — reco audit n°1.
- **Tests end-to-end** du pipeline complet (`ingest → clean → train → calibrate
  → predict`) sur dataset synthétique hermétique (aucun réseau, disque et MLflow
  isolés) — reco audit n°2.
- **Manifeste de traçabilité SHA-256** des artefacts de données
  (`data/manifest.json`, versionné) : hash, dimensions, producteur, date et SHA
  git du code ; auto-enregistrement par `ingest`/`clean` avec rattrapage des
  artefacts préexistants ; CLI `readmission-data-verify` (exit 1 sur dérive,
  utilisable en CI) — reco audit n°3.
- **Authentification par clé API** (en-tête `X-API-Key`, comparaison
  `hmac.compare_digest` anti-timing) sur les endpoints d'ops, et **rate limiting**
  à fenêtre glissante par IP (429 + `Retry-After`) — reco audit n°4.
- **Monitoring online des prédictions** : journal SQLite privacy-by-design
  (sorties + latence uniquement, jamais les features) et endpoint
  `GET /monitoring/summary` (volume, distribution du risque, latences,
  incertitude conformelle, borne d'erreur attendue) — reco audit n°6.
- **Onglet « Monitoring production »** dans la démo Streamlit (client urllib
  stdlib, état dégradé gracieux, secrets `MONITORING_API_URL`/`MONITORING_API_KEY`).
- **Endpoint `GET /ready`** : readiness réelle (modèle chargé + prédiction
  factice + mémoire RSS, 503 sinon), séparée de la liveness `/health` — reco
  audit n°10.

### Modifié

- `/predict` est **public par design** (démo testable par un recruteur) et protégé
  par le rate limiting ; la clé API ne protège que les endpoints d'ops
  (`/monitoring/summary`). La capacité d'auth sur `/predict` reste documentée
  (une ligne) pour un contexte de données de santé réelles.
- `Dockerfile` : uvicorn lancé avec `--proxy-headers` (IP client réelle derrière
  nginx, requis pour le rate limiting).
- `.dockerignore` : exclusion du `.env` (injection via `--env-file` au runtime).
- `docs/deployment.md` : `--env-file` + volume `readmission-data` (persistance
  du journal de monitoring).
- README : badge 85 tests, sections Sécurité / Sondes / Monitoring / Qualité /
  Structure réécrites, diagramme d'architecture aligné.

## [1.0.0] - 2026-07-16

État présenté à l'audit externe (note 18,5/20) — pipeline ML de bout en bout,
pensé pour la production, construit en 13 étapes documentées dans
`journal/journaldebord.md`.

### Ajouté

- **Données** : ingestion reproductible (UCI Diabetes 130-US, ~101 k séjours),
  contrat Pandera, nettoyage avec neutralisation des fuites de cible
  (décès/soins palliatifs), regroupement clinique des codes ICD-9.
- **Validation anti-fuite triple** : split patient-disjoint (StratifiedGroupKFold
  sur `patient_nbr`), retrait des cibles mécaniquement déterminées, préprocesseur
  ajusté dans chaque fold de CV.
- **Modélisation** : baseline régression logistique, LightGBM tuné par Optuna
  (TPE, 25 essais, PR-AUC), tracking MLflow (SQLite).
- **Confiance** : calibration des probabilités (isotonic/sigmoïde, Brier
  0,225 → 0,097), conformal prediction MAPIE (couverture cible 90 %, empirique
  ~0,906) avec découpage fit/calib/conform patient-disjoint.
- **Explicabilité & équité** : SHAP global et local, audit par sous-groupes
  (sexe, âge, race), Model Card, Decision Curve Analysis (`dcurves`, MSKCC).
- **MLOps** : API FastAPI (`/health`, `/predict`), Docker, CI GitHub Actions
  (ruff + pytest), monitoring de drift Evidently avec règle de ré-entraînement,
  démo Streamlit, déploiement production sur VPS OVHcloud (nginx, TLS, UFW,
  fail2ban) — 4,49 € HT/mois.

[1.1.0]: https://github.com/behramkorkut/readmission-risk-ml/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/behramkorkut/readmission-risk-ml/releases/tag/v1.0.0

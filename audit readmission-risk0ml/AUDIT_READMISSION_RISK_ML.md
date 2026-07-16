# Audit du projet `readmission-risk-ml`

**Projet :** Prédiction du risque de réadmission hospitalière à 30 jours (patients diabétiques)  
**Auteur :** Behram Korkut  
**Date de l'audit :** Juillet 2026  
**Répertoire audité :** `/Users/behramko/Postule_folder/Data_Engineer_apply/readmission-risk-ml/`

---

## Résumé exécutif

Ce projet est un **pipeline Machine Learning de bout en bout, pensé pour la production**, démontrant une maîtrise exceptionnelle du cycle de vie ML appliqué à la santé. Il se distingue par une rigueur méthodologique rare dans les projets portfolio (triple anti-fuite, validation groupée par patient), une approche moderne de la confiance (calibration + conformal prediction), et un déploiement réel sur cloud souverain français. Le niveau global est **senior / staff engineer** sur un profil Data Science / MLOps.

**Note globale : 18,5 / 20**

---

## 1. Architecture & Structure du projet

### ✅ Forces

- **Layout `src/` professionnel** : le paquet `readmission_risk` est séparé des tests et des applications, conforme aux standards Python modernes. L'utilisation de `uv` comme gestionnaire de dépendances et build system est avant-gardiste (2026).
- **Modularité par domaine** : découpage clair en `data/`, `validation/`, `features/`, `modeling/`, `evaluation/`, `serving/`, `monitoring/`, `common/`. Chaque module a une responsabilité unique et est réutilisable.
- **CLI reproductible** : 9 commandes déclarées dans `pyproject.toml` (`readmission-ingest`, `clean`, `train-baseline`, `train-gboost`, `calibrate`, `explain`, `dca`, `drift`, `serve`) qui couvrent tout le pipeline.
- **Configuration centralisée** : `pydantic-settings` avec `.env`, graine fixe, chemins typés. Un seul endroit pour modifier la configuration.
- **Gestion des dépendances** : `uv.lock` garantit la reproductibilité exacte des environnements. Deux groupes (`dev`, `demo`) bien séparés.

### ⚠️ Faiblesses / Axes d'amélioration

- **Dossier `notebooks/` vide** : l'absence de notebooks d'exploration EDA est notable. Bien que le journal de bord compense pédagogiquement, un notebook d'analyse exploratoire (profilage, corrélations, visualisations) serait un plus pour la reproductibilité visuelle et la communication.
- **Pas de gestion de versions de données (DVC)** : les données ne sont pas versionnées. En production, `data/diabetic_clean.parquet` pourrait être régénéré sans traçabilité de la version du code qui l'a produit.
- **Structure MLflow locale** : le backend SQLite (`mlflow.db`) et le stockage de fichiers local (`mlruns/`) conviennent pour un projet solo, mais ne passeraient pas à l'échelle en équipe. Un serveur MLflow distant serait nécessaire.

---

## 2. Rigueur Méthodologique (Validation & Anti-fuite)

### ✅ Forces — **Niveau excellence**

- **Triple anti-fuite rigoureuse** :
  1. **Split patient-disjoint** (`StratifiedGroupKFold` sur `patient_nbr`) : empêche qu'un même patient apparaisse à la fois en train et en test. Le garde-fou `assert_no_group_overlap` est un excellent réflexe défensif.
  2. **Retrait des fuites de cible** : suppression des séjours "décès/soins palliatifs" dont la cible est mécaniquement déterminée. Subtilité que beaucoup de tutoriaux sur ce dataset ratent.
  3. **Preprocessing intra-CV** : le `ColumnTransformer` est ajusté **dans chaque fold**, jamais sur tout le dataset. C'est l'erreur la plus fréquente en data science et elle est ici parfaitement évitée.
- **Découpage à 3 jeux pour calibration/conformal** : `fit` / `calib` / `conform` tous patient-disjoints. C'est une rigueur quasi-académique rarement vue dans des projets industriels.
- **Métriques adaptées au déséquilibre** : PR-AUC comme métrique primaire (base ~0,114), ROC-AUC secondaire, Brier pour la calibration. L'accuracy n'est pas utilisée — excellent.

### ⚠️ Faiblesses / Axes d'amélioration

- **Pas de cross-validation imbriquée (nested CV) pour le tuning** : bien que la CV à 3 folds pendant le tuning soit honnête, une CV imbriquée (outer + inner) donnerait une estimation encore moins biaisée de la performance en généralisation. C'est un détail technique mais pertinent pour une revue académique.
- **Test hold-out unique** : un seul split 80/20. Pour une évaluation encore plus robuste, on pourrait itérer sur plusieurs splits hold-out et moyenner.

---

## 3. Modélisation & Performance

### ✅ Forces

- **Baseline logistique avant complexe** : bon réflexe méthodologique. La baseline (PR-AUC 0,215) sert de référence honnête.
- **LightGBM + Optuna (TPE)** : état de l'art pour le tabulaire. 25 essais bayésiens optimisant directement la PR-AUC. Le gain relatif de +7 % est réaliste et honnête.
- **Gestion du déséquilibre** : `scale_pos_weight` et `class_weight='balanced'` sont utilisés de manière cohérente. Le compromis calibration vs ranking est bien compris et corrigé à l'étape 7.
- **Plafond de performance documenté** : PR-AUC ~0,23 est modeste mais l'auteur le contextualise parfaitement (base de hasard 0,114, fourchette publiée 0,64-0,68 en ROC-AUC). Il ne survente pas ses résultats.

### ⚠️ Faiblesses / Axes d'amélioration

- **Hyperparamètres figés dans le code** : `DEFAULT_LGBM_PARAMS` est codé en dur dans `gboost.py`. En production, ces paramètres devraient être chargés dynamiquement depuis MLflow Model Registry ou un fichier de config externe.
- **Pas d'exploration de modèles alternatifs** : pas de test de XGBoost, CatBoost, ou de modèles linéaires régularisés plus sophistiqués. Pour un projet portfolio, c'est acceptable ; en production, un benchmark plus large serait attendu.
- **Pas de feature selection** : 234 features après encodage sans sélection (RFE, LASSO, importance). Des features redondantes pourraient complexifier inutilement le modèle.

---

## 4. Confiance, Explicabilité & Utilité Clinique

### ✅ Forces — **Différenciateurs majeurs**

- **Calibration des probabilités** : comparaison isotonic vs sigmoïde sur un jeu de calibration indépendant. Brier divisé par ~2 (0,195 → 0,097). C'est critique en santé où un "30 %" doit vraiment signifier 30 %.
- **Conformal Prediction (MAPIE)** : utilise `SplitConformalClassifier` avec garantie de couverture à 90 %. L'adaptateur `DataFrameAdapter` custom pour interfacer MAPIE avec le pipeline sklearn est élégant et bien pensé. La couverture empirique de ~0,906 prouve que la garantie est tenue.
- **SHAP global + local** : `TreeExplainer` sur le modèle de base. Les graphiques beeswarm, bar et local sont tous produits. L'API calcule les top raisons SHAP à la volée.
- **Decision Curve Analysis (DCA)** : évaluation de l'utilité clinique via `dcurves` (MSKCC). Le modèle domine les stratégies naïves sur toute la plage de seuils cliniquement plausibles. C'est un **réflexe de haut niveau** rarement vu même chez des data scientists expérimentés.
- **Audit d'équité par sous-groupes** : analyse par sexe, âge et race avec distinction soignée entre "vrai biais" (âge) et "bruit d'échantillon" (race). Le seuil `MIN_GROUP_SIZE=200` est pertinent.

### ⚠️ Faiblesses / Axes d'amélioration

- **Taille moyenne d'ensemble conformel = 1,067** : très proche de 1, ce qui signifie que l'ensemble est souvent un singleton. C'est une force (le modèle est confiant) mais aussi une faiblesse : l'incertitude n'est pas souvent exprimée comme "je ne sais pas" (`{non, réadmission}`). En pratique clinique, on aimerait peut-être une couverture plus conservative (95 %) pour obtenir plus d'ensembles doubles quand le modèle hésite.
- **Pas de conformal Mondrian** : l'audit d'équité mentionne le conformal Mondrian dans le journal de bord (étape 8) mais il n'est pas implémenté. Cela permettrait de garantir la couverture **par sous-groupe**, ce qui serait puissant pour l'équité.
- **SHAP sur échantillon de test** : le calcul SHAP global est fait sur un échantillon de 2000 patients (pour la vitesse). C'est raisonnable mais introduit une variabilité.

---

## 5. MLOps, Production & Déploiement

### ✅ Forces — **Niveau production**

- **API FastAPI déployée en production** : endpoints `/health` et `/predict` avec validation Pydantic, documentation Swagger auto-générée. Le modèle est chargé une seule fois (cache `_STATE`).
- **Déploiement cloud souverain français** : VPS OVHcloud à Strasbourg, 4,49 €/mois. Durcissement manuel (SSH clés, UFW, fail2ban, nginx reverse proxy, TLS Let's Encrypt). C'est un **engagement réel** qui dépasse largement le "déploiement sur Heroku" classique des portfolios.
- **Docker** : image basée sur `ghcr.io/astral-sh/uv`, modèle embarqué, runtime OpenMP pour LightGBM. Build en 2 couches (cache).
- **CI/CD GitHub Actions** : lint (ruff) + tests (pytest) à chaque push/PR. Le workflow installe le runtime OpenMP pour LightGBM — détail bien géré.
- **Monitoring de drift (Evidently)** : injection contrôlée de dérive + rapport HTML interactif. Règle de décision : >20 % de colonnes driftées → ré-entraînement recommandé.
- **Streamlit Cloud** : démo interactive avec profils pré-remplis, affichage du risque calibré, ensemble conformel et raisons SHAP.
- **MLflow tracking** : params, métriques et artefacts tracés pour chaque étape clé (baseline, gboost, calibration, explicabilité).

### ⚠️ Faiblesses / Axes d'amélioration

- **Pas de test d'intégration HTTP** : les tests API appellent directement les fonctions Python plutôt que via `TestClient` de FastAPI. Cela ne teste pas la sérialisation JSON, la gestion des headers, ni les codes HTTP. Un test `TestClient` serait un plus.
- **Pas d'authentification sur l'API** : l'endpoint `/predict` est public (ce qui est OK pour une démo mais pas pour des données de santé réelles). En production, une clé API ou OAuth2 serait obligatoire.
- **Pas de monitoring des prédictions en temps réel** : le monitoring actuel ne regarde que le drift des features (distribution). Il manque le tracking de la distribution des prédictions, du taux d'erreur estimé, et des latences.
- **Pas de Health Check avancé** : le endpoint `/health` vérifie seulement la présence du fichier modèle. Il ne teste pas le chargement réussi, ni la mémoire, ni la capacité à faire une prédiction de test.
- **Pas de rate limiting** : l'API n'a pas de limitation de débit, ce qui la rend vulnérable à un usage abusif.
- **MLflow local** : pas de Model Registry utilisé pour la gestion des versions de modèles. Le modèle est embarqué dans l'image Docker via `models/model.joblib` — pratique simple mais sans gestion de cycle de vie des modèles.

---

## 6. Qualité du Code & Tests

### ✅ Forces

- **33 tests pytest** couvrant : schéma, nettoyage, split, features, CV, gboost, calibration, API, explicabilité, drift, utilité clinique. Couverture fonctionnelle très complète.
- **Tests basés sur des propriétés** : les tests DCA vérifient des invariants théoriques ("none" = 0, "all" suit la formule, modèle parfait domine) plutôt que des valeurs magiques. C'est un signe de maturité.
- **Lint ruff** : configuration propre (`line-length=120`, `target-version=py312`, sélecteurs E/F/I/UP/B).
- **Docstrings** : présentes dans les modules clés, expliquant le "pourquoi" et non seulement le "quoi".
- **Imports paresseux** : `mlflow` et `shap` importés à l'intérieur des fonctions `main()` pour éviter les dépendances lourdes lors des tests unitaires.
- **Logging structuré** : `structlog` utilisé dans tous les modules pour des logs exploitables.
- **Gestion des types** : annotations de type `from __future__ import annotations`, utilisation de `pandas-styled` typage.

### ⚠️ Faiblesses / Axes d'amélioration

- **Tests API en direct** : comme mentionné, pas de test via `TestClient` FastAPI. Le test `test_api.py` appelle `health()` et `predict()` directement, ce qui contourne la couche HTTP.
- **Pas de tests end-to-end du pipeline complet** : il n'y a pas de test qui exécute `ingest → clean → train → calibrate → predict` en chaîne. Un tel test garantirait que les modules fonctionnent ensemble.
- **Docstrings vides** : certains `__init__.py` sont vides. Les fichiers de tests pourraient avoir des docstrings décrivant ce qu'ils testent (test de quelle propriété).
- **Pas de type hints stricts** : quelques endroits utilisent `Any` (ex: `dict[str, Any]` dans l'API). `TypedDict` ou un modèle Pydantic plus strict seraient préférables pour les features d'entrée.

---

## 7. Documentation & Communication

### ✅ Forces — **Excellence**

- **README portfolio-ready** : problème clair, résultats chiffrés (tableau comparatif), architecture Mermaid, pipeline en commandes CLI, exemple d'appel API curl, badges. C'est un README de niveau "open source popular".
- **Model Card** : document standard d'IA responsable couvrant usage prévu, hors-périmètre, données, performances, équité, limites, considérations éthiques (RGPD/HDS). Très valorisé en santé.
- **Journal de bord pédagogique** (`journal/journaldebord.md`, 647 lignes) : explique chaque étape avec le "quoi", le "pourquoi" et le "comment". C'est un atout majeur pour la reproductibilité pédagogique.
- **Documentation de déploiement** (`docs/deployment.md`) : procédure complète avec pièges rencontrés et exploitation courante.
- **Architecture en diagrammes Mermaid** : le README utilise des diagrammes de flux pour expliquer le pipeline et l'anti-fuite.

### ⚠️ Faiblesses / Axes d'amélioration

- **Pas de changelog** : pas de `CHANGELOG.md` ou de versions sémantiques pour suivre l'évolution du projet (utile pour un projet qui a 13 étapes).
- **Pas de documentation API autre que Swagger** : pas de guide d'intégration pour les consommateurs de l'API (code exemple en Python, gestion d'erreurs, limites de rate).

---

## 8. Éthique, Équité & Responsabilité

### ✅ Forces — **Niveau exemplaire**

- **Humain dans la boucle explicité** : la Model Card stipule clairement que c'est un outil d'aide, pas un décideur, et qu'il ne doit pas être utilisé pour rationner les soins.
- **Audit d'équité** : analyse par sexe, âge et race. Le biais âge est **documenté honnêtement** (dégradation sur les patients très âgés) plutôt que caché.
- **Données dé-identifiées** : utilisation d'un dataset public déjà anonymisé (UCI). Mention de la conformité RGPD/HDS pour un déploiement sur données réelles.
- **Limites clairement énoncées** : données anciennes (1999-2008) et américaines, non directement transférables. Plafond de prédictibilité bas. C'est de l'honnêteté intellectuelle rare.

### ⚠️ Faiblesses / Axes d'amélioration

- **Pas de mécanisme de recours** : en cas de décision contestable, il n'y a pas de processus documenté pour qu'un patient ou un clinicien puisse demander une révision ou une explication complémentaire.
- **Pas d'évaluation du risque d'usage dual** : bien que le hors-périmètre soit défini, il n'y a pas d'analyse des risques d'utilisation malveillante (ex: refus de couverture par une assurance).

---

## Tableau récapitulatif

| Domaine | Note | Commentaire |
|---------|:----:|-------------|
| Architecture & Structure | 18/20 | Layout pro, CLI, config centralisée. Manque DVC et notebooks EDA. |
| Rigueur Méthodologique | **20/20** | Triple anti-fuite, split patient-disjoint, preprocessing intra-CV. Niveau académique. |
| Modélisation & Performance | 17/20 | Baseline honnête, Optuna, LightGBM. HP figés, pas de feature selection. |
| Confiance & Explicabilité | **20/20** | Calibration + Conformal + SHAP + DCA + équité. Différenciateurs majeurs. |
| MLOps & Production | 17/20 | API déployée, Docker, CI/CD, drift. Manque auth, rate limiting, monitoring prédiction. |
| Qualité du Code & Tests | 17/20 | 33 tests, ruff, structlog. Manque tests HTTP et E2E. |
| Documentation & Communication | **19/20** | README, Model Card, journal pédagogique. Manque changelog et guide API. |
| Éthique & Responsabilité | **19/20** | Humain dans la boucle, audit équité, limites documentées. |
| **Total** | **18,5/20** | **Projet portfolio de très haut niveau, senior+.** |

---

## Recommandations prioritaires (par ordre d'impact)

### 🔴 Haute priorité — Pour viser un poste Staff/Principal DS

1. **Ajouter des tests d'intégration HTTP** avec `TestClient` FastAPI pour valider la sérialisation JSON, les codes HTTP et la gestion d'erreurs.
2. **Ajouter un test end-to-end** du pipeline complet (`ingest → clean → train → calibrate → predict`) pour garantir la cohérence entre les modules.
3. **Versionner les données avec DVC** (ou au minimum documenter le hash MD5 des Parquets) pour la traçabilité.
4. **Ajouter de l'authentification à l'API** (même simple, type clé API en header) et du rate limiting.

### 🟡 Moyenne priorité — Pour renforcer la crédibilité production

5. **Implémenter le conformal Mondrian** par sous-groupe (âge, sexe) pour garantir la couverture équitable.
6. **Ajouter un monitoring des prédictions** (distribution des scores, latence, taux d'erreur estimé) en complément du drift des features.
7. **Charger les hyperparamètres depuis MLflow Model Registry** au lieu du code dur.
8. **Ajouter un notebook EDA** dans `notebooks/` avec profilage, corrélations et visualisations des données brutes.

### 🟢 Basse priorité — Polish

9. **Ajouter un `CHANGELOG.md`** et tagger les versions sémantiques (`v1.0.0`).
10. **Rendre le endpoint `/health` plus robuste** : test de prédiction factice, vérification mémoire.
11. **Explorer la feature selection** (LASSO, RFE) pour réduire les 234 features.

---

## Verdict final

Ce projet est un **exemple de portfolio Data Science / MLOps de très haut niveau**. Il démontre non seulement des compétences techniques solides (Python, sklearn, LightGBM, FastAPI, Docker) mais surtout une **maturité méthodologique et éthique** rare :

- La rigueur de la validation anti-fuite est irréprochable.
- L'attention portée à la calibration et à l'incertitude (conformal prediction) est un vrai différenciateur.
- La DCA prouve que l'auteur pense en termes d'**utilité métier**, pas seulement de métriques ML.
- Le déploiement réel sur OVHcloud français montre un **engagement opérationnel** au-delà du code.

Les faiblesses identifiées sont des **axes d'amélioration** pour viser un poste senior/staff, pas des failles critiques. Le projet est déjà **prêt à être présenté à un recruteur** et suffisamment solide pour servir de base à une discussion technique approfondie.

**Recommandation :** ✅ **Projet exemplaire — à valoriser fortement dans les entretiens.**

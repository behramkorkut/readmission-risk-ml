# Model Card — Readmission Risk 30 jours

## Détails du modèle
- **Type** : gradient boosting **LightGBM** (tuné par Optuna), avec **calibration des
  probabilités** (isotonic) et **conformal prediction** (MAPIE, LAC, couverture 90 %).
- **Tâche** : classification binaire — risque de réadmission hospitalière à moins de 30
  jours pour des patients diabétiques.
- **Sortie** : probabilité calibrée + ensemble de prédiction conformel + 5 raisons SHAP.
- **Version** : 1.0.

## Usage prévu
- **Aide à la décision** : prioriser les interventions post-sortie (suivi renforcé,
  éducation thérapeutique) sur les patients à risque élevé.
- **Utilisateurs** : équipes cliniques et de coordination des soins, avec un humain dans
  la boucle.
- **Hors périmètre** : ⚠️ ce modèle n'est **pas** destiné à des décisions cliniques
  automatiques, ni à refuser/limiter des soins. C'est un outil d'aide, pas un décideur.

## Données
- **Source** : *Diabetes 130-US hospitals (1999-2008)*, UCI ML Repository — ~101 766
  séjours, 50 variables, données hospitalières dé-identifiées (États-Unis).
- **Cible** : `readmitted_30d` (1 si réadmission < 30 j). Prévalence ~11 %.
- **Nettoyage** : retrait des séjours décès/soins palliatifs (fuite de cible), regroupement
  des codes ICD-9 en catégories cliniques, imputation/encodage.

## Méthodologie d'évaluation
- **Anti-fuite** : split et validation croisée **groupés par patient** (StratifiedGroupKFold) ;
  un patient n'est jamais à la fois en entraînement et en test.
- **Calibration/conformal** : découpage à 3 jeux patient-disjoints (fit / calib / conform),
  évaluation finale sur un **test hold-out** jamais vu.

## Performances (test hold-out)
| Métrique | Valeur | Référence |
|----------|--------|-----------|
| PR-AUC (primaire) | ~0,23 | base de hasard 0,114 |
| ROC-AUC | ~0,67 | fourchette publiée 0,64-0,68 |
| Brier (calibré) | ~0,097 | vs ~0,195 non calibré |
| Couverture conformelle | ~0,90 | cible 0,90 |

## Équité (audit par sous-groupes)
- **Sexe** : équitable (écart de ROC-AUC ≈ 0,002).
- **Âge** : **biais réel** — le modèle se dégrade sur les patients très âgés (ROC-AUC
  [90-100] ≈ 0,60 vs ≈ 0,80 chez les plus jeunes). À surveiller / documenter.
- **Origine** : écart apparent dû à de **petits échantillons** (bruit), pas un biais
  systématique sur les grands groupes.

## Limites
- **Plafond de prédictibilité bas** : la réadmission est multifactorielle ; la performance
  est modeste *par nature* sur ce problème.
- **Données anciennes et états-uniennes (1999-2008)** : non transférables telles quelles à
  un hôpital français/actuel → nécessiteraient un **ré-entraînement et une recalibration**
  sur des données locales et récentes.
- **Dérive** : la performance se dégrade si la population change → **monitoring de drift**
  et ré-entraînement requis (voir `readmission-drift`).

## Considérations éthiques
- Humain dans la boucle obligatoire ; ne pas utiliser pour rationner les soins.
- En cas de déploiement sur données réelles : conformité **RGPD/HDS**, minimisation,
  traçabilité, et réévaluation régulière de l'équité.

## Comment reproduire
Voir le `README.md` (séquence `readmission-ingest` → `clean` → `train-gboost` →
`calibrate` → `explain` → `drift` → `serve`).

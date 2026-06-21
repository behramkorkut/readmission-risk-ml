"""Baseline : régression logistique + validation croisée par patient + MLflow.

Donne le premier chiffre de référence honnête, auquel on comparera le gradient
boosting (étape 6). La CV se fait sur le TRAIN ; le jeu de test (hold-out) reste
intact, réservé à l'évaluation finale.
"""

from __future__ import annotations

import mlflow
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from readmission_risk.common.config import settings
from readmission_risk.data.loaders import load_clean
from readmission_risk.data.split import make_holdout_split
from readmission_risk.features.build import build_feature_spec, build_preprocessor
from readmission_risk.modeling.cv import grouped_cv_scores, summarize

log = structlog.get_logger()


def build_baseline_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Préprocesseur + régression logistique (class_weight='balanced' pour le déséquilibre)."""
    pre = build_preprocessor(numeric, categorical)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # compense les ~11 % de positifs
        random_state=settings.random_seed,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


def main() -> None:
    df = load_clean()
    train, _test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )

    numeric, categorical, constant = build_feature_spec(train, settings.target_col)
    pipe = build_baseline_pipeline(numeric, categorical)

    X = train[numeric + categorical]
    y = train[settings.target_col]
    groups = train[settings.patient_id_col]

    log.info("baseline.cv_start", n_features_raw=len(numeric) + len(categorical), n_rows=len(train))
    scores = grouped_cv_scores(pipe, X, y, groups, n_splits=5, seed=settings.random_seed)
    summary = summarize(scores)

    print("\n=== Baseline (régression logistique) — CV 5 folds groupés par patient ===")
    print(f"PR-AUC  : {summary['pr_auc_mean']:.4f} ± {summary['pr_auc_std']:.4f}  (primaire ; base = {y.mean():.3f})")
    print(f"ROC-AUC : {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")
    print(f"Brier   : {summary['brier_mean']:.4f} ± {summary['brier_std']:.4f}  (plus bas = mieux)")

    # --- Suivi MLflow ---
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name="baseline-logreg"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "n_numeric": len(numeric),
            "n_categorical": len(categorical),
            "n_constant_dropped": len(constant),
            "cv": "StratifiedGroupKFold(5)",
        })
        mlflow.log_metrics(summary)
    log.info("baseline.done", **{k: round(v, 4) for k, v in summary.items()})


if __name__ == "__main__":
    main()

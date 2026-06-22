"""Gradient boosting (LightGBM) + tuning Optuna, comparé à la baseline.

Stratégie :
- Tuning des hyperparamètres avec Optuna (recherche bayésienne TPE) en CV groupée
  par patient à 3 folds, en optimisant la PR-AUC (notre métrique primaire).
- Évaluation finale du meilleur jeu d'hyperparamètres en CV 5 folds (mêmes folds
  que la baseline -> comparaison équitable).
- scale_pos_weight pour compenser le déséquilibre (~11 % de positifs).
Tout est tracé dans MLflow pour comparer proprement au run baseline.
"""

from __future__ import annotations

import optuna
import structlog
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from readmission_risk.common.config import settings
from readmission_risk.data.loaders import load_clean
from readmission_risk.data.split import make_holdout_split
from readmission_risk.features.build import build_feature_spec, build_preprocessor
from readmission_risk.modeling.cv import grouped_cv_scores, summarize

log = structlog.get_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence le bruit d'Optuna

# Paramètres LightGBM fixes (non tunés).
# n_jobs=1 : on parallélise plutôt les folds de CV (évite la sur-souscription threads).
FIXED_PARAMS = dict(objective="binary", n_jobs=1, verbose=-1, subsample_freq=1)


def build_pipeline(numeric, categorical, params: dict, scale_pos_weight: float) -> Pipeline:
    pre = build_preprocessor(numeric, categorical)
    clf = LGBMClassifier(
        **FIXED_PARAMS,
        random_state=settings.random_seed,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


def _suggest(trial: optuna.Trial) -> dict:
    """Espace de recherche des hyperparamètres LightGBM."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def tune(X, y, groups, numeric, categorical, scale_pos_weight, n_trials, folds, seed) -> optuna.Study:
    """Lance l'étude Optuna : maximise la PR-AUC moyenne en CV groupée."""

    def objective(trial: optuna.Trial) -> float:
        pipe = build_pipeline(numeric, categorical, _suggest(trial), scale_pos_weight)
        scores = grouped_cv_scores(pipe, X, y, groups, n_splits=folds, seed=seed)
        return float(scores["test_pr_auc"].mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def main() -> None:
    df = load_clean()
    train, _test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )
    numeric, categorical, _ = build_feature_spec(train, settings.target_col)
    X = train[numeric + categorical]
    y = train[settings.target_col]
    groups = train[settings.patient_id_col]
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())

    log.info(
        "gboost.tune_start",
        n_trials=settings.lgbm_n_trials,
        tuning_folds=settings.tuning_cv_folds,
        scale_pos_weight=round(scale_pos_weight, 2),
    )
    study = tune(
        X, y, groups, numeric, categorical, scale_pos_weight,
        settings.lgbm_n_trials, settings.tuning_cv_folds, settings.random_seed,
    )
    best = study.best_params

    # Évaluation finale en 5 folds (comparable à la baseline).
    final_pipe = build_pipeline(numeric, categorical, best, scale_pos_weight)
    scores = grouped_cv_scores(final_pipe, X, y, groups, n_splits=5, seed=settings.random_seed)
    summary = summarize(scores)

    print("\n=== LightGBM (tuné Optuna) — CV 5 folds groupés par patient ===")
    print(f"PR-AUC  : {summary['pr_auc_mean']:.4f} ± {summary['pr_auc_std']:.4f}  (baseline = 0.2151)")
    print(f"ROC-AUC : {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}  (baseline = 0.6639)")
    print(f"Brier   : {summary['brier_mean']:.4f} ± {summary['brier_std']:.4f}  (baseline = 0.2250)")
    print(f"\nMeilleurs hyperparamètres : {best}")

    import mlflow  # import paresseux : non requis par build_pipeline/tune

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name="lightgbm-optuna"):
        mlflow.log_params({
            "model": "LightGBM",
            "scale_pos_weight": round(scale_pos_weight, 3),
            "n_trials": settings.lgbm_n_trials,
            "tuning_folds": settings.tuning_cv_folds,
            **best,
        })
        mlflow.log_metrics(summary)
        mlflow.log_metric("tuning_best_pr_auc", float(study.best_value))
    log.info("gboost.done", **{k: round(v, 4) for k, v in summary.items()})


if __name__ == "__main__":
    main()

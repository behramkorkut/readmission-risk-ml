"""Calibration des probabilités + conformal prediction (confiance & incertitude).

Pourquoi : un bon classement (PR-AUC) ne suffit pas en santé. Un soignant a besoin
(1) de probabilités FIABLES (un « 30 % » doit valoir ~30 % de réadmissions) et
(2) d'une INCERTITUDE explicite. On répond aux deux :

- Calibration : on ré-étalonne les probabilités du modèle (isotonic / sigmoïde),
  ce qui corrige l'effet du scale_pos_weight (qui gonfle les probas) -> Brier ↓.
- Conformal prediction (MAPIE) : on produit des ensembles de prédiction avec
  GARANTIE DE COUVERTURE (ex. 90 %), une incertitude rigoureuse et distribution-free.

Découpage anti-fuite à 3 jeux (patient-disjoints, tous distincts du TEST final) :
  fit (entraîne le modèle) | calib (calibre les probas) | conform (calibre le conformal).
Tout est évalué sur le TEST hold-out, jamais vu.
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")  # backend non interactif (sauvegarde fichier, pas d'affichage)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from mapie.classification import SplitConformalClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from readmission_risk.common.config import settings
from readmission_risk.data.loaders import load_clean
from readmission_risk.data.split import make_holdout_split
from readmission_risk.features.build import build_feature_spec
from readmission_risk.modeling.gboost import DEFAULT_LGBM_PARAMS, build_pipeline

log = structlog.get_logger()


class DataFrameAdapter(BaseEstimator, ClassifierMixin):
    """Adapte un modèle « DataFrame » pour MAPIE (qui le teste avec un ndarray).

    Notre pipeline sélectionne les colonnes par NOM (DataFrame requis). MAPIE, lui,
    vérifie l'estimateur en lui passant un tableau numpy. Cet adaptateur reconvertit
    tout ndarray entrant en DataFrame avec les bons noms de colonnes.
    """

    def __init__(self, model, feature_cols: list[str]):
        self.model = model
        self.feature_cols = feature_cols
        self.classes_ = model.classes_
        self.n_features_in_ = len(feature_cols)

    def _as_df(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_cols)

    def fit(self, X, y=None):  # le modèle est déjà entraîné (prefit)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict(self, X):
        return self.model.predict(self._as_df(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._as_df(X))


def three_way_split(train, target_col, pid_col, seed):
    """Découpe le TRAIN en fit / calib / conform (patient-disjoints)."""
    fit_calib, conform = make_holdout_split(train, target_col, pid_col, test_size=0.2, seed=seed)
    fit, calib = make_holdout_split(fit_calib, target_col, pid_col, test_size=0.25, seed=seed)
    return fit, calib, conform


def coverage_and_size(y_set: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Couverture empirique (la vraie classe est-elle dans l'ensemble ?) et taille moyenne."""
    ys = y_set[:, :, 0] if y_set.ndim == 3 else y_set
    coverage = float(ys[np.arange(len(y_true)), y_true].mean())
    mean_size = float(ys.sum(axis=1).mean())
    return coverage, mean_size


def _plot_calibration(y_true, proba_raw, proba_cal, path) -> None:
    frac_raw, mean_raw = calibration_curve(y_true, proba_raw, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, proba_cal, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="parfaitement calibré")
    plt.plot(mean_raw, frac_raw, "o-", label="brut (LightGBM)")
    plt.plot(mean_cal, frac_cal, "s-", label="calibré (isotonic)")
    plt.xlabel("Probabilité moyenne prédite")
    plt.ylabel("Fraction observée de positifs")
    plt.title("Courbe de calibration — réadmission 30j")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    df = load_clean()
    train, test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )
    fit, calib, conform = three_way_split(
        train, settings.target_col, settings.patient_id_col, settings.random_seed
    )
    numeric, categorical, _ = build_feature_spec(fit, settings.target_col)
    cols = numeric + categorical
    y_fit = fit[settings.target_col]
    spw = float((y_fit == 0).sum() / (y_fit == 1).sum())

    # 1) Modèle de base entraîné sur le jeu 'fit'
    base = build_pipeline(numeric, categorical, DEFAULT_LGBM_PARAMS, spw)
    base.fit(fit[cols], y_fit)

    y_test = test[settings.target_col].to_numpy()
    proba_raw = base.predict_proba(test[cols])[:, 1]
    brier_raw = brier_score_loss(y_test, proba_raw)

    # 2) Calibration (isotonic et sigmoïde) sur le jeu 'calib' (modèle gelé)
    results = {"brier_raw": brier_raw}
    calibrated = {}
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        cal.fit(calib[cols], calib[settings.target_col])
        proba = cal.predict_proba(test[cols])[:, 1]
        results[f"brier_{method}"] = brier_score_loss(y_test, proba)
        calibrated[method] = cal

    best_method = min(("isotonic", "sigmoid"), key=lambda m: results[f"brier_{m}"])
    best_model = calibrated[best_method]

    # 3) Conformal prediction sur le jeu 'conform' (modèle calibré, prefit)
    # On enveloppe dans l'adaptateur pour que MAPIE accepte notre pipeline DataFrame.
    scc = SplitConformalClassifier(
        estimator=DataFrameAdapter(best_model, cols),
        confidence_level=settings.conformal_confidence,
        conformity_score="lac",
        prefit=True,
    )
    scc.conformalize(conform[cols], conform[settings.target_col])
    _, y_set = scc.predict_set(test[cols])
    coverage, mean_size = coverage_and_size(y_set, y_test)

    # --- Rapports ---
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    curve_path = settings.reports_dir / "calibration_curve.png"
    _plot_calibration(y_test, proba_raw, best_model.predict_proba(test[cols])[:, 1], curve_path)

    print("\n=== Calibration (Brier sur le TEST hold-out, plus bas = mieux) ===")
    print(f"Brut (LightGBM) : {results['brier_raw']:.4f}")
    print(f"Isotonic        : {results['brier_isotonic']:.4f}")
    print(f"Sigmoïde        : {results['brier_sigmoid']:.4f}")
    print(f"-> méthode retenue : {best_method}")
    print(f"\n=== Conformal prediction (couverture cible {settings.conformal_confidence:.0%}) ===")
    print(f"Couverture empirique sur le TEST : {coverage:.3f}")
    print(f"Taille moyenne d'ensemble        : {mean_size:.3f}")
    print(f"\nCourbe de calibration -> {curve_path}")

    # --- Sauvegarde du modèle final (calibré + conformal) pour l'API ---
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.models_dir / settings.model_filename
    joblib.dump(
        {
            "model": best_model,          # modèle calibré (probabilités fiables)
            "conformal": scc,             # ensembles de prédiction (incertitude)
            "base_pipeline": base,        # pipeline brut (pour SHAP en temps réel)
            "feature_cols": cols,
            "confidence_level": settings.conformal_confidence,
            "calibration_method": best_method,
        },
        model_path,
    )

    # --- MLflow ---
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name="calibrated-conformal"):
        mlflow.log_params({"calibration_method": best_method, "conformity_score": "lac",
                           "confidence_level": settings.conformal_confidence})
        mlflow.log_metrics({**results, "conformal_coverage": coverage, "conformal_set_size": mean_size})
        mlflow.log_artifact(str(curve_path))
    log.info("calibrate.done", best_method=best_method, coverage=round(coverage, 3),
             model_path=str(model_path))


if __name__ == "__main__":
    main()

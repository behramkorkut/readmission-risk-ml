"""Explicabilité (SHAP) + audit d'équité par sous-groupes.

- SHAP : pourquoi le modèle prédit ce qu'il prédit. Importance GLOBALE (quelles
  variables pèsent le plus) et LOCALE (pourquoi CE patient est à risque).
  Indispensable pour l'acceptation clinique d'un modèle.
- Équité : performance et taux d'alerte par sous-groupes (sexe, âge, origine).
  Détecte d'éventuels biais — réflexe attendu en santé.

SHAP s'applique sur le LightGBM de base (arbres) : TreeExplainer attribue à chaque
feature sa contribution. On l'applique sur la matrice transformée par le préprocesseur.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import average_precision_score, roc_auc_score

from readmission_risk.common.config import settings
from readmission_risk.data.loaders import load_clean
from readmission_risk.data.split import make_holdout_split
from readmission_risk.features.build import build_feature_spec
from readmission_risk.modeling.gboost import DEFAULT_LGBM_PARAMS, build_pipeline

log = structlog.get_logger()

SUBGROUPS = ["gender", "age", "race"]
MIN_GROUP_SIZE = 200  # on n'évalue les métriques que sur des groupes assez grands


def subgroup_metrics(
    df: pd.DataFrame, target_col: str, proba: np.ndarray, subgroups: list[str]
) -> pd.DataFrame:
    """Calcule par sous-groupe : effectif, taux réel de positifs, risque moyen prédit, ROC-AUC."""
    rows = []
    for col in subgroups:
        for value, sub in df.groupby(col, observed=True):
            y = sub[target_col].to_numpy()
            p = proba[sub.index.to_numpy()]
            auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
            rows.append({
                "attribut": col,
                "groupe": str(value),
                "n": len(sub),
                "taux_reel": round(float(y.mean()), 3),
                "risque_moyen_predit": round(float(p.mean()), 3),
                "roc_auc": round(float(auc), 3) if not np.isnan(auc) else None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    import shap  # import lourd -> paresseux

    df = load_clean()
    train, test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )
    numeric, categorical, _ = build_feature_spec(train, settings.target_col)
    cols = numeric + categorical
    y_train = train[settings.target_col]
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    pipe = build_pipeline(numeric, categorical, DEFAULT_LGBM_PARAMS, spw)
    pipe.fit(train[cols], y_train)
    pre, lgbm = pipe.named_steps["prep"], pipe.named_steps["clf"]
    feat_names = list(pre.get_feature_names_out())

    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    # --- SHAP sur un échantillon de test (vitesse) ---
    sample = test.sample(min(settings.shap_sample_size, len(test)), random_state=settings.random_seed)
    X_sample = pre.transform(sample[cols])
    X_dense = X_sample.toarray() if hasattr(X_sample, "toarray") else np.asarray(X_sample)

    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_dense)
    if isinstance(shap_values, list):  # certaines versions renvoient une liste par classe
        shap_values = shap_values[1]

    # Importance globale (bar)
    shap.summary_plot(shap_values, X_dense, feature_names=feat_names, plot_type="bar",
                      max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(settings.reports_dir / "shap_global.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Effets détaillés (beeswarm)
    shap.summary_plot(shap_values, X_dense, feature_names=feat_names, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(settings.reports_dir / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Explication LOCALE : le patient au risque prédit le plus élevé
    proba_sample = pipe.predict_proba(sample[cols])[:, 1]
    idx = int(np.argmax(proba_sample))
    contrib = shap_values[idx]
    top = np.argsort(np.abs(contrib))[-10:]
    plt.figure(figsize=(8, 5))
    plt.barh([feat_names[i] for i in top], [contrib[i] for i in top],
             color=["#c0392b" if contrib[i] > 0 else "#2980b9" for i in top])
    plt.axvline(0, color="k", lw=0.8)
    plt.title(f"Contributions SHAP — patient à risque {proba_sample[idx]:.0%}")
    plt.xlabel("Impact sur le log-odds (rouge = augmente le risque)")
    plt.tight_layout()
    plt.savefig(settings.reports_dir / "shap_local.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- Audit d'équité ---
    proba_test = pipe.predict_proba(test[cols])[:, 1]
    fairness = subgroup_metrics(test, settings.target_col, proba_test, SUBGROUPS)
    fairness_path = settings.reports_dir / "fairness.csv"
    fairness.to_csv(fairness_path, index=False)

    print("\n=== Top features (importance SHAP globale) -> reports/shap_global.png ===")
    print("\n=== Audit d'équité (groupes n >=", MIN_GROUP_SIZE, ") ===")
    big = fairness[fairness["n"] >= MIN_GROUP_SIZE]
    print(big.to_string(index=False))
    # Écart de ROC-AUC entre groupes (par attribut) = indicateur de biais.
    print("\nÉcart de ROC-AUC max-min par attribut :")
    for attr in SUBGROUPS:
        aucs = big[(big["attribut"] == attr)]["roc_auc"].dropna()
        if len(aucs) >= 2:
            print(f"  {attr:<8} : {aucs.max() - aucs.min():.3f}")

    # --- MLflow ---
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name="explain-fairness"):
        for art in ("shap_global.png", "shap_beeswarm.png", "shap_local.png", "fairness.csv"):
            mlflow.log_artifact(str(settings.reports_dir / art))
        ap = average_precision_score(test[settings.target_col], proba_test)
        mlflow.log_metric("test_pr_auc", float(ap))
    log.info("explain.done", reports=str(settings.reports_dir))


if __name__ == "__main__":
    main()

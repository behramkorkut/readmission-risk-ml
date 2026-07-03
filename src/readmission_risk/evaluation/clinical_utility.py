"""Utilité clinique : courbe de décision (Decision Curve Analysis, Vickers & Elkin).

Pourquoi : PR-AUC dit si le modèle CLASSE bien, le Brier dit si les probabilités
sont FIABLES — mais aucun des deux ne répond à la question clinique : « à partir
de quel seuil de risque déclencher un suivi post-sortie, et le modèle fait-il
mieux que les politiques naïves ? ». La DCA compare le BÉNÉFICE NET du modèle à
deux stratégies de référence : suivre tout le monde (« all ») et ne suivre
personne (« none »), en pénalisant les faux positifs selon le seuil choisi :

    NB(t) = TP/n − FP/n · t/(1−t)

Calcul délégué au package de référence `dcurves` (MSKCC). Évaluation sur le
TEST hold-out patient-disjoint, avec le modèle final calibré — les mêmes
conditions que le reste de l'évaluation.
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")  # backend non interactif (sauvegarde fichier)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from dcurves import dca

from readmission_risk.common.config import settings
from readmission_risk.data.loaders import load_clean
from readmission_risk.data.split import make_holdout_split

log = structlog.get_logger()

# Seuils de décision cliniquement plausibles (au-delà de 50 %, personne ne
# déciderait un programme de suivi sur ce type de risque).
THRESHOLDS = np.arange(0.01, 0.51, 0.01)


def compute_decision_curve(
    y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray = THRESHOLDS
) -> pd.DataFrame:
    """Bénéfice net du modèle vs « all » vs « none », par seuil de décision.

    Renvoie le DataFrame long de `dcurves` : colonnes `model` (model/all/none),
    `threshold`, `net_benefit`.
    """
    data = pd.DataFrame(
        {"outcome": np.asarray(y_true).astype(int), "model": np.asarray(proba, dtype=float)}
    )
    return dca(data=data, outcome="outcome", modelnames=["model"], thresholds=thresholds)


def summarize(curve: pd.DataFrame, at: tuple[float, ...] = (0.10, 0.15, 0.20)) -> pd.DataFrame:
    """Bénéfice net aux seuils clés (lecture rapide pour le rapport / la CI)."""
    rows = []
    for t in at:
        sub = curve[np.isclose(curve["threshold"], t)]
        rows.append({
            "seuil": t,
            **{m: float(sub.loc[sub["model"] == m, "net_benefit"].iloc[0])
               for m in ("model", "all", "none")},
        })
    return pd.DataFrame(rows)


def _plot(curve: pd.DataFrame, path) -> None:
    styles = {
        "all": ("#d62728", 1.2, "Suivre tout le monde"),
        "none": ("#7f7f7f", 1.2, "Ne suivre personne"),
        "model": ("#1f77b4", 2.2, "LightGBM calibré"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (color, lw, label) in styles.items():
        sub = curve[curve["model"] == name]
        ax.plot(sub["threshold"], sub["net_benefit"], color=color, lw=lw, label=label)
    ax.set_xlabel("Seuil de décision (probabilité de réadmission)")
    ax.set_ylabel("Bénéfice net")
    ax.set_title("Courbe de décision — suivi post-sortie (test hold-out)")
    ax.set_ylim(-0.02, 0.14)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    # Mêmes données, même split, même modèle que l'évaluation finale.
    df = load_clean()
    _, test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )
    bundle = joblib.load(settings.models_dir / settings.model_filename)
    proba = bundle["model"].predict_proba(test[bundle["feature_cols"]])[:, 1]

    curve = compute_decision_curve(test[settings.target_col].to_numpy(), proba)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "decision_curve.png"
    _plot(curve, out)

    table = summarize(curve)
    print("\n=== Bénéfice net (test hold-out, patient-disjoint) ===")
    print(table.round(4).to_string(index=False))
    print(f"\nCourbe de décision -> {out}")
    log.info("dca_done", report=str(out))


if __name__ == "__main__":
    main()
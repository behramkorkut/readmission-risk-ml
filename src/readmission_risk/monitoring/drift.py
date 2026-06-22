"""Monitoring : détection de dérive des données (Evidently) + déclencheur de ré-entraînement.

En production, la population évolue (vieillissement, nouveaux protocoles, changement
de codage…) : le modèle voit des données différentes de celles d'entraînement, ses
performances se dégradent silencieusement. On détecte ça en comparant la distribution
des features entre un jeu de RÉFÉRENCE (train) et les données COURANTES.

Pour démontrer le mécanisme, on injecte une dérive contrôlée et réaliste, on génère un
rapport Evidently, et on applique une règle de décision de ré-entraînement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from readmission_risk.validation.schema import AGE_BUCKETS

log = structlog.get_logger()


def inject_drift(df: pd.DataFrame, seed: int = 0, intensity: float = 1.0) -> pd.DataFrame:
    """Simule une dérive réaliste : population plus lourde, vieillissante, + de manquants."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    # 1) Dérive numérique : une population plus "lourde" touche TOUTES les variables de
    #    sévérité (séjours, actes, médicaments, recours antérieurs, comorbidités).
    def _bump(col, factor=None, add_max=None):
        if col not in out:
            return
        s = out[col].astype(float)
        if factor:
            s = s * (1 + factor * intensity)
        if add_max:
            s = s + rng.integers(0, add_max + 1, n) * intensity
        out[col] = s.round().astype(int)

    _bump("time_in_hospital", add_max=3)
    _bump("num_lab_procedures", factor=0.2)
    _bump("num_procedures", add_max=2)
    _bump("num_medications", factor=0.3)
    _bump("number_outpatient", add_max=1)
    _bump("number_emergency", add_max=1)
    _bump("number_inpatient", add_max=1)
    _bump("number_diagnoses", add_max=2)
    if "time_in_hospital" in out:
        out["time_in_hospital"] = out["time_in_hospital"].clip(1, 14)
    if "number_diagnoses" in out:
        out["number_diagnoses"] = out["number_diagnoses"].clip(1, 16)

    # 2) Dérive catégorielle : vieillissement (une partie des patients monte d'une tranche d'âge)
    if "age" in out:
        idx = {b: i for i, b in enumerate(AGE_BUCKETS)}
        bump = rng.random(n) < 0.25 * intensity
        new_age = out["age"].map(idx).to_numpy()
        new_age[bump] = np.minimum(new_age[bump] + 1, len(AGE_BUCKETS) - 1)
        out["age"] = [AGE_BUCKETS[i] for i in new_age]

    # 3) Diabète moins bien contrôlé : davantage de A1Cresult > 8
    if "A1Cresult" in out:
        out.loc[rng.random(n) < 0.3 * intensity, "A1Cresult"] = ">8"

    # 4) Dérive de qualité : davantage de medical_specialty manquant
    if "medical_specialty" in out:
        out.loc[rng.random(n) < 0.2 * intensity, "medical_specialty"] = np.nan

    return out


def run_drift_report(reference, current, numeric, categorical, out_html) -> tuple[float, int]:
    """Génère le rapport Evidently (HTML) et renvoie (part de colonnes driftées, nb)."""
    data_def = DataDefinition(numerical_columns=list(numeric), categorical_columns=list(categorical))
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current, data_definition=data_def)
    snapshot = Report(metrics=[DataDriftPreset()]).run(reference_data=ref_ds, current_data=cur_ds)
    snapshot.save_html(str(out_html))
    return _extract_drift(snapshot.dict())


def _extract_drift(report_dict: dict) -> tuple[float, int]:
    """Extrait la part et le nombre de colonnes driftées du rapport."""
    for metric in report_dict.get("metrics", []):
        value = metric.get("value")
        if isinstance(value, dict) and "share" in value:
            return float(value["share"]), int(value.get("count", 0))
    return float("nan"), 0


def retraining_decision(drift_share: float, threshold: float) -> tuple[bool, str]:
    """Règle de déclenchement : au-delà du seuil, on recommande un ré-entraînement."""
    if drift_share > threshold:
        return True, (
            f"DÉRIVE SIGNIFICATIVE ({drift_share:.0%} de colonnes > seuil {threshold:.0%}) "
            "-> déclencher un ré-entraînement."
        )
    return False, f"Dérive sous le seuil ({drift_share:.0%} <= {threshold:.0%}) -> pas d'action."


def main() -> None:
    from readmission_risk.common.config import settings
    from readmission_risk.data.loaders import load_clean
    from readmission_risk.data.split import make_holdout_split
    from readmission_risk.features.build import build_feature_spec

    df = load_clean()
    train, test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )
    numeric, categorical, _ = build_feature_spec(train, settings.target_col)
    cols = numeric + categorical

    k = min(settings.monitoring_sample, len(test))
    reference = train[cols].sample(k, random_state=settings.random_seed)
    current = inject_drift(test[cols].sample(k, random_state=settings.random_seed), seed=settings.random_seed)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out_html = settings.reports_dir / "drift_report.html"
    share, count = run_drift_report(reference, current, numeric, categorical, out_html)
    trigger, message = retraining_decision(share, settings.drift_threshold)

    print("\n=== Monitoring de dérive (référence = train, courant = test + dérive injectée) ===")
    print(f"Colonnes driftées : {count}/{len(cols)}  (part = {share:.1%})")
    print(message)
    print(f"Rapport détaillé  -> {out_html}")

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    with mlflow.start_run(run_name="drift-monitoring"):
        mlflow.log_metrics({"drift_share": share, "drifted_columns": count})
        mlflow.log_param("retraining_triggered", trigger)
        mlflow.log_artifact(str(out_html))
    log.info("drift.done", drift_share=round(share, 3), trigger=trigger)


if __name__ == "__main__":
    main()

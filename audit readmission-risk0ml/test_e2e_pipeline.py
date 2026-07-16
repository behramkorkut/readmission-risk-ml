"""Tests end-to-end du pipeline complet (recommandation d'audit — haute priorité n°2).

On exécute la VRAIE chaîne, module par module, sur un dataset synthétique fidèle
au schéma UCI Diabetes 130-US :

    ingest -> validation du schéma brut -> clean -> train (Optuna réduit)
           -> calibrate (+ conformal) -> API (/health, /predict)

Verrous d'isolation :
- aucun accès réseau (le téléchargement UCI est mocké) ;
- tout le disque dans tmp_path (data / models / reports) et cwd déplacé, donc
  aucune écriture parasite (MLflow, mlruns) dans le dépôt ;
- MLflow redirigé vers une SQLite temporaire ;
- tuning Optuna réduit (2 essais x 2 folds) pour rester rapide.

On ne vérifie PAS la performance du modèle (sans sens sur du synthétique) mais la
COHÉRENCE INTER-MODULES : contrats d'artefacts produits/consommés, colonnes servies
par l'API = colonnes produites par le nettoyage, anti-fuite patient respectée sur
toute la chaîne, et l'API qui sert bien le modèle issu de l'étape de calibration.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from readmission_risk.common.config import settings
from readmission_risk.data import clean as clean_mod
from readmission_risk.data import ingest as ingest_mod
from readmission_risk.data.clean import DEATH_HOSPICE_DISPOSITIONS
from readmission_risk.data.split import assert_no_group_overlap, make_holdout_split
from readmission_risk.modeling import calibrate as calibrate_mod
from readmission_risk.modeling import gboost as gboost_mod
from readmission_risk.modeling.calibrate import three_way_split
from readmission_risk.serving import api
from readmission_risk.validation.schema import AGE_BUCKETS, validate_raw

# Pool de codes ICD-9 réalistes (diabète, cardiaque, respiratoire, etc.).
DIAG_POOL = ["250.83", "250.01", "250.6", "428", "414", "486", "276", "599", "786", "V45"]


def _synthetic_raw(n_patients: int = 400, seed: int = 7) -> pd.DataFrame:
    """Dataset brut synthétique respectant le contrat de `validation/schema.py`.

    Le risque de réadmission croît avec l'historique d'hospitalisations, pour que
    la chaîne apprenne un vrai signal (sinon la calibration dégénère). Quelques
    séjours « décès » (discharge 11) et 2 genres « Unknown/Invalid » permettent de
    vérifier que le nettoyage filtre bien ce qu'il prétend filtrer.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    encounter_id = 100_000
    for p in range(n_patients):
        patient_nbr = 5_000_000 + p
        frailty = float(rng.normal())  # hétérogénéité patient -> structure de groupes réelle
        for stay in range(int(rng.integers(1, 3))):  # 1 ou 2 séjours par patient
            n_inpatient = int(rng.poisson(0.7))
            n_emergency = int(rng.poisson(0.4))
            logit = -2.0 + 0.8 * n_inpatient + 0.5 * n_emergency + 0.4 * frailty
            p_pos = float(1.0 / (1.0 + np.exp(-logit)))
            u = float(rng.random())
            readmitted = "<30" if u < p_pos else (">30" if u < p_pos + 0.35 else "NO")
            death = bool(rng.random() < 0.03)
            invalid = p < 2 and stay == 0  # 2 séjours à genre invalide (filtrés par clean)
            rows.append(
                {
                    "encounter_id": encounter_id,
                    "patient_nbr": patient_nbr,
                    "race": str(
                        rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "?"], p=[0.7, 0.2, 0.05, 0.05])
                    ),
                    "gender": "Unknown/Invalid" if invalid else str(rng.choice(["Male", "Female"])),
                    "age": str(rng.choice(AGE_BUCKETS[3:])),  # [30-40) .. [90-100)
                    "weight": "?" if rng.random() < 0.97 else "[50-75)",
                    "admission_type_id": int(rng.choice([1, 2, 3, 6])),
                    "discharge_disposition_id": 11 if death else int(rng.choice([1, 3, 6])),
                    "admission_source_id": int(rng.choice([1, 2, 7])),
                    "time_in_hospital": int(rng.integers(1, 15)),  # 1..14 (contrat schéma)
                    "payer_code": "?" if rng.random() < 0.5 else str(rng.choice(["MC", "HM", "BC"])),
                    "medical_specialty": str(
                        rng.choice(["InternalMedicine", "Cardiology", "Family/GeneralPractice", "?"])
                    ),
                    "num_lab_procedures": int(rng.poisson(40)),
                    "num_procedures": int(rng.poisson(2)),
                    "num_medications": int(rng.integers(1, 25)),
                    "number_outpatient": int(rng.poisson(0.5)),
                    "number_emergency": n_emergency,
                    "number_inpatient": n_inpatient,
                    "number_diagnoses": int(rng.integers(1, 10)),
                    "max_glu_serum": str(rng.choice(["None", "Norm", ">200", ">300"], p=[0.8, 0.1, 0.05, 0.05])),
                    "A1Cresult": str(rng.choice(["None", "Norm", ">7", ">8"], p=[0.8, 0.1, 0.05, 0.05])),
                    "metformin": str(rng.choice(["No", "Steady", "Up", "Down"], p=[0.7, 0.2, 0.05, 0.05])),
                    "insulin": str(rng.choice(["No", "Steady", "Up", "Down"], p=[0.5, 0.3, 0.1, 0.1])),
                    "change": str(rng.choice(["Ch", "No"])),
                    "diabetesMed": str(rng.choice(["Yes", "No"], p=[0.8, 0.2])),
                    "diag_1": str(rng.choice(DIAG_POOL)),
                    "diag_2": "?" if rng.random() < 0.2 else str(rng.choice(DIAG_POOL)),
                    "diag_3": "?" if rng.random() < 0.4 else str(rng.choice(DIAG_POOL)),
                    "readmitted": readmitted,
                }
            )
            encounter_id += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Exécute la chaîne réelle UNE fois pour tous les tests du module.

    Toute erreur dans un maillon (ingest, clean, train, calibrate) fait échouer
    cette fixture — c'est déjà un test : la chaîne doit tourner de bout en bout.
    """
    tmp = tmp_path_factory.mktemp("e2e")
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp)  # aucune écriture relative (mlruns/, artefacts) dans le dépôt
    mp.setattr(settings, "data_dir", tmp / "data")
    mp.setattr(settings, "models_dir", tmp / "models")
    mp.setattr(settings, "reports_dir", tmp / "reports")
    mp.setattr(settings, "mlflow_tracking_uri", f"sqlite:///{tmp}/mlflow_e2e.db")
    mp.setattr(settings, "lgbm_n_trials", 2)  # tuning réel mais réduit
    mp.setattr(settings, "tuning_cv_folds", 2)

    raw = _synthetic_raw()
    mp.setattr(ingest_mod, "download_raw", lambda: raw.copy())  # aucun réseau

    # 1) ingest -> parquet brut ; 2) le contrat de données tient sur la chaîne réelle
    raw_path = ingest_mod.ingest(force=True)
    validate_raw(pd.read_parquet(raw_path))

    # 3) clean -> parquet nettoyé ; 4) train (Optuna réduit) ; 5) calibrate (+ conformal)
    clean_mod.main()
    gboost_mod.main()
    calibrate_mod.main()

    yield {
        "raw": raw,
        "clean": pd.read_parquet(settings.data_dir / settings.clean_filename),
        "model_path": settings.models_dir / settings.model_filename,
    }
    api._STATE.clear()
    mp.undo()


def test_clean_contracts(pipeline_run):
    """Le nettoyage tient ses contrats sur la chaîne réelle (filtrages + cible)."""
    df, raw = pipeline_run["clean"], pipeline_run["raw"]
    # Filtrages : décès/soins palliatifs + genres invalides, ni plus ni moins.
    # L'oracle est recalculé depuis le brut (un séjour peut cumuler les deux motifs).
    leak = raw["discharge_disposition_id"].isin(DEATH_HOSPICE_DISPOSITIONS)
    invalid = raw["gender"] == "Unknown/Invalid"
    assert len(df) == len(raw) - int((leak | invalid).sum())
    assert not df["discharge_disposition_id"].isin(DEATH_HOSPICE_DISPOSITIONS).any()
    # Cible binaire saine, et les deux classes survivent au filtrage.
    assert df[settings.target_col].isin([0, 1]).all()
    assert df[settings.target_col].nunique() == 2
    # Colonnes attendues : cible 3-classes retirée, poids retiré, groupes ICD-9 créés.
    assert "readmitted" not in df.columns
    assert "weight" not in df.columns
    assert {"diag_1_group", "diag_2_group", "diag_3_group"} <= set(df.columns)
    assert not df[settings.patient_id_col].isna().any()  # clé anti-fuite intacte


def test_trained_artifact_coherence(pipeline_run):
    """L'artefact servi est bien produit par la chaîne et aligné sur la table nettoyée."""
    assert pipeline_run["model_path"].exists()
    bundle = joblib.load(pipeline_run["model_path"])
    expected = {"model", "conformal", "base_pipeline", "feature_cols", "confidence_level", "calibration_method"}
    assert expected <= set(bundle)
    # Les colonnes attendues par l'API existent réellement dans la sortie de clean.
    available = set(pipeline_run["clean"].columns) - {"encounter_id", settings.patient_id_col, settings.target_col}
    assert set(bundle["feature_cols"]) <= available
    assert bundle["calibration_method"] in {"isotonic", "sigmoid"}
    assert bundle["confidence_level"] == settings.conformal_confidence


def test_patient_disjointness_along_chain(pipeline_run):
    """Anti-fuite n°1 vérifiée de bout en bout sur la donnée réellement nettoyée."""
    df = pipeline_run["clean"]
    train, test = make_holdout_split(
        df, settings.target_col, settings.patient_id_col, settings.test_size, settings.random_seed
    )
    assert_no_group_overlap(train, test, settings.patient_id_col)
    fit, calib, conform = three_way_split(train, settings.target_col, settings.patient_id_col, settings.random_seed)
    assert_no_group_overlap(fit, calib, settings.patient_id_col)
    assert_no_group_overlap(fit, conform, settings.patient_id_col)
    assert_no_group_overlap(calib, conform, settings.patient_id_col)
    for subset in (fit, calib, conform):  # le test final ne fuite dans aucun sous-jeu
        assert_no_group_overlap(subset, test, settings.patient_id_col)


def test_api_serves_chain_output(pipeline_run):
    """L'API charge l'artefact produit par la chaîne et rend une prédiction cohérente."""
    api._STATE.clear()
    assert api.health() == {"status": "ok", "model_loaded": True}
    payload = {  # profil patient réaliste (format de l'endpoint /predict)
        "age": "[60-70)",
        "gender": "Female",
        "race": "Caucasian",
        "time_in_hospital": 7,
        "num_medications": 15,
        "number_inpatient": 2,
        "number_emergency": 1,
        "number_diagnoses": 8,
        "discharge_disposition_id": 1,
        "admission_type_id": 1,
        "insulin": "Steady",
        "diabetesMed": "Yes",
        "change": "Ch",
        "diag_1_group": "Diabetes",
    }
    resp = api.predict(api.PredictRequest(features=payload))
    assert 0.0 <= resp.risk <= 1.0
    assert resp.risk_label in set(api.LABELS.values())
    assert resp.prediction_set and set(resp.prediction_set) <= set(api.LABELS.values())
    assert len(resp.top_reasons) == 5
    assert all(r.direction in {"augmente", "diminue"} for r in resp.top_reasons)
    assert resp.confidence_level == pytest.approx(settings.conformal_confidence)
    api._STATE.clear()

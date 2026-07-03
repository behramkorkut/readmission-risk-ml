"""Démo interactive — Readmission Risk (Streamlit).

Score de risque calibré + ensemble de prédiction conformel + raisons SHAP,
sur le modèle produit par `readmission-calibrate` (models/model.joblib).

La logique de prédiction est identique à l'API FastAPI (serving/api.py) ;
cette app n'est qu'une vitrine interactive au-dessus du même bundle.

Déploiement Streamlit Cloud : pointer l'app sur app/streamlit_app.py (Python 3.12),
dépendances dans requirements.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Le bundle contient un adaptateur défini dans readmission_risk.modeling.calibrate :
# le paquet doit être importable pour dé-sérialiser.
ROOT = Path(__file__).resolve().parents[1]  # racine du repo (app/ est un sous-dossier)
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "model.joblib"
LABELS = {0: "non réadmis", 1: "réadmission < 30 j"}
PREVALENCE = 0.114  # taux de base de la classe positive (test hold-out)

st.set_page_config(page_title="Readmission Risk — démo", page_icon="🏥", layout="wide")


# ---------- Chargement (une seule fois) ----------
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_bundle() -> dict[str, Any]:
    import shap  # import différé : coûteux

    bundle = joblib.load(MODEL_PATH)
    prep = bundle["base_pipeline"].named_steps["prep"]
    return {
        **bundle,
        "prep": prep,
        "explainer": shap.TreeExplainer(bundle["base_pipeline"].named_steps["clf"]),
        "feat_names": list(prep.get_feature_names_out()),
    }


def pretty_feature(name: str) -> str:
    """'cat__insulin_Up' -> 'insulin = Up', 'num__number_inpatient' -> 'number_inpatient'."""
    name = name.split("__", 1)[-1]
    return name


def predict(state: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Même logique que l'endpoint /predict de l'API."""
    row = {col: features.get(col, np.nan) for col in state["feature_cols"]}
    df = pd.DataFrame([row], columns=state["feature_cols"])

    proba = float(state["model"].predict_proba(df)[:, 1][0])

    _, y_set = state["conformal"].predict_set(df)
    members = y_set[0, :, 0] if y_set.ndim == 3 else y_set[0]
    prediction_set = [LABELS[i] for i, inside in enumerate(members) if inside]

    x = state["prep"].transform(df)
    x_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    sv = state["explainer"].shap_values(x_dense)
    if isinstance(sv, list):
        sv = sv[1]
    contrib = np.asarray(sv)[0]
    top_idx = np.argsort(np.abs(contrib))[-6:][::-1]
    reasons = [
        {"feature": pretty_feature(state["feat_names"][i]), "shap": float(contrib[i])}
        for i in top_idx
        if abs(contrib[i]) > 1e-6
    ]
    return {"risk": proba, "prediction_set": prediction_set, "reasons": reasons}


# ---------- Profils pré-remplis ----------
PROFILES = {
    "— Personnaliser —": {},
    "Profil bas risque": {
        "age": "[40-50)", "number_inpatient": 0, "number_emergency": 0,
        "number_outpatient": 0, "time_in_hospital": 2, "num_medications": 8,
        "number_diagnoses": 4, "num_lab_procedures": 35, "insulin": "No",
        "diabetesMed": "No", "change": "No", "diag_1_group": "Musculoskeletal",
        "discharge_disposition_id": 1, "admission_type_id": 3,
    },
    "Profil haut risque": {
        "age": "[80-90)", "number_inpatient": 6, "number_emergency": 4,
        "number_outpatient": 3, "time_in_hospital": 12, "num_medications": 28,
        "number_diagnoses": 9, "num_lab_procedures": 75, "insulin": "Up",
        "diabetesMed": "Yes", "change": "Ch", "diag_1_group": "Circulatory",
        "discharge_disposition_id": 22, "admission_type_id": 1,
    },
}

ADMISSION_TYPES = {1: "1 — Urgence", 2: "2 — Urgent", 3: "3 — Électif", 5: "5 — Non renseigné", 6: "6 — Inconnu"}
DISCHARGES = {
    1: "1 — Retour au domicile",
    3: "3 — Établissement de soins infirmiers (SNF)",
    6: "6 — Domicile avec soins à domicile",
    2: "2 — Transfert court séjour",
    22: "22 — Transfert réadaptation",
    5: "5 — Autre établissement",
}

# ---------- UI ----------
st.title("Readmission Risk — démo interactive")
st.markdown(
    "Prédiction du risque de **réadmission hospitalière à moins de 30 jours** (patients diabétiques, "
    "dataset UCI *Diabetes 130-US hospitals*, ~100 k séjours). "
    "Probabilités **calibrées** (isotonic), incertitude **garantie** par conformal prediction (90 %), "
    "explications **SHAP** par prédiction. "
    "[Code & démarche](https://github.com/behramkorkut/readmission-risk-ml) · "
    "[Model card](https://github.com/behramkorkut/readmission-risk-ml/blob/main/MODEL_CARD.md)"
)

with st.sidebar:
    st.header("Dossier patient")
    profile = st.selectbox("Profil pré-rempli", list(PROFILES.keys()))
    d = PROFILES[profile]

    def dv(key, default):  # valeur du profil sinon défaut
        return d.get(key, default)

    ages = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    age = st.select_slider("Âge", ages, value=dv("age", "[60-70)"))
    gender = st.radio("Sexe", ["Female", "Male"], horizontal=True)

    st.subheader("Historique (12 derniers mois)")
    number_inpatient = st.slider("Hospitalisations antérieures", 0, 10, dv("number_inpatient", 0),
                                 help="Variable la plus prédictive (SHAP)")
    number_emergency = st.slider("Passages aux urgences", 0, 10, dv("number_emergency", 0))
    number_outpatient = st.slider("Consultations externes", 0, 10, dv("number_outpatient", 0))

    st.subheader("Séjour actuel")
    time_in_hospital = st.slider("Durée du séjour (jours)", 1, 14, dv("time_in_hospital", 4))
    num_medications = st.slider("Médicaments administrés", 1, 40, dv("num_medications", 15))
    number_diagnoses = st.slider("Diagnostics posés", 1, 16, dv("number_diagnoses", 7))
    num_lab_procedures = st.slider("Actes de laboratoire", 1, 100, dv("num_lab_procedures", 45))
    diag_1_group = st.selectbox("Diagnostic principal", ["Circulatory", "Respiratory", "Digestive",
                                "Diabetes", "Injury", "Musculoskeletal", "Genitourinary",
                                "Neoplasms", "Other"],
                                index=["Circulatory", "Respiratory", "Digestive", "Diabetes",
                                       "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms",
                                       "Other"].index(dv("diag_1_group", "Circulatory")))
    admission_type_id = st.selectbox("Type d'admission", list(ADMISSION_TYPES),
                                     format_func=ADMISSION_TYPES.get,
                                     index=list(ADMISSION_TYPES).index(dv("admission_type_id", 1)))
    discharge_disposition_id = st.selectbox("Sortie vers", list(DISCHARGES),
                                            format_func=DISCHARGES.get,
                                            index=list(DISCHARGES).index(dv("discharge_disposition_id", 1)))

    st.subheader("Traitement diabète")
    insulin = st.selectbox("Insuline", ["No", "Steady", "Up", "Down"],
                           index=["No", "Steady", "Up", "Down"].index(dv("insulin", "No")))
    diabetesMed = st.radio("Antidiabétique prescrit", ["Yes", "No"], horizontal=True,
                           index=["Yes", "No"].index(dv("diabetesMed", "Yes")))
    change = st.radio("Changement de traitement", ["No", "Ch"], horizontal=True,
                      index=["No", "Ch"].index(dv("change", "No")))
    a1c = st.selectbox("HbA1c (A1Cresult)", ["Non mesurée", "Norm", ">7", ">8"])

features = {
    "age": age, "gender": gender,
    "number_inpatient": number_inpatient, "number_emergency": number_emergency,
    "number_outpatient": number_outpatient, "time_in_hospital": time_in_hospital,
    "num_medications": num_medications, "number_diagnoses": number_diagnoses,
    "num_lab_procedures": num_lab_procedures, "diag_1_group": diag_1_group,
    "admission_type_id": admission_type_id, "discharge_disposition_id": discharge_disposition_id,
    "insulin": insulin, "diabetesMed": diabetesMed, "change": change,
}
if a1c != "Non mesurée":
    features["A1Cresult"] = a1c
# Les colonnes non renseignées sont imputées par le pipeline (comme dans l'API).

state = load_bundle()
res = predict(state, features)
risk = res["risk"]

# ---------- Résultats ----------
col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Risque calibré")
    st.metric("Probabilité de réadmission < 30 j", f"{risk:.1%}",
              delta=f"{risk - PREVALENCE:+.1%} vs taux de base ({PREVALENCE:.1%})",
              delta_color="inverse")
    st.progress(min(risk, 1.0))
    if risk >= 2 * PREVALENCE:
        st.error("Risque élevé — patient à prioriser pour un suivi post-sortie.")
    elif risk >= PREVALENCE:
        st.warning("Risque supérieur au taux de base.")
    else:
        st.success("Risque inférieur au taux de base.")

    st.subheader("Incertitude (conformal, 90 %)")
    pset = res["prediction_set"]
    st.write("Ensemble de prédiction : " + " · ".join(f"`{m}`" for m in pset))
    if len(pset) == 1:
        st.info("Le modèle **tranche** : une seule classe dans l'ensemble, avec 90 % de couverture garantie.")
    else:
        st.info("Le modèle **ne tranche pas** : les deux classes restent plausibles. "
                "C'est une information clinique en soi — ce cas mérite un regard humain.")
    st.caption("Probabilité calibrée par régression isotonique (Brier 0,225 → 0,097). "
               "Ensemble conformel : couverture empirique ~90 % sur le test hold-out patient-disjoint.")

with col2:
    st.subheader("Pourquoi ? (SHAP)")
    reasons = pd.DataFrame(res["reasons"])
    if not reasons.empty:
        reasons["effet"] = np.where(reasons["shap"] > 0, "augmente le risque", "diminue le risque")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.2))
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in reasons["shap"]]
        ax.barh(reasons["feature"][::-1], reasons["shap"][::-1],
                color=colors[::-1])
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlabel("Contribution SHAP (log-odds)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Rouge : pousse vers la réadmission · Vert : la rend moins probable. "
                   "Contributions du modèle LightGBM sous-jacent (avant calibration).")

st.divider()
st.markdown(
    "⚠️ **Démo portfolio, pas un dispositif médical.** Données 1999-2008, biais documentés "
    "(voir la [model card](https://github.com/behramkorkut/readmission-risk-ml/blob/main/MODEL_CARD.md)) : "
    "audit d'équité OK selon le sexe, dégradation connue sur les patients très âgés."
)

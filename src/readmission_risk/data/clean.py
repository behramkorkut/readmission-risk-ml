"""Nettoyage des données brutes + neutralisation des fuites.

Trois préoccupations distinctes, dans l'ordre :
1. Sentinelle de manquant : remplacer la chaîne « ? » par de vrais NaN.
2. Anti-fuite de cible : retirer les séjours dont l'issue rend la réadmission
   impossible (décès, soins palliatifs) — sinon le modèle « triche ».
3. Mise en forme : cible binaire, regroupement clinique des codes ICD-9
   (des centaines de niveaux → ~9 catégories exploitables).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

MISSING_TOKEN = "?"

# discharge_disposition_id correspondant à un décès ou à des soins palliatifs :
# 11 = décédé · 13/14 = hospice · 19/20/21 = décédé (hospice Medicaid).
# Ces séjours ne PEUVENT pas être suivis d'une réadmission -> fuite de cible.
DEATH_HOSPICE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]


def replace_missing_token(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace la sentinelle « ? » par NaN (manquant exploitable par sklearn)."""
    return df.replace(MISSING_TOKEN, np.nan)


def drop_leakage_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les séjours décès / soins palliatifs (réadmission impossible)."""
    mask = df["discharge_disposition_id"].isin(DEATH_HOSPICE_DISPOSITIONS)
    return df.loc[~mask].copy()


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crée readmitted_30d (1 si « <30 », sinon 0) et retire la cible 3-classes."""
    df = df.copy()
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype("int8")
    return df.drop(columns=["readmitted"])


def _icd9_group(code) -> str:
    """Regroupe un code ICD-9 en grande catégorie clinique (cf. Strack et al. 2014)."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):  # codes V/E = circonstances, non maladies
        return "Other"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 250 <= num < 251:
        return "Diabetes"
    if (390 <= num <= 459) or num == 785:
        return "Circulatory"
    if (460 <= num <= 519) or num == 786:
        return "Respiratory"
    if (520 <= num <= 579) or num == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if (580 <= num <= 629) or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def group_icd9(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace diag_1/2/3 (codes bruts) par diag_1/2/3_group (catégories)."""
    df = df.copy()
    for col in DIAG_COLS:
        df[col + "_group"] = df[col].map(_icd9_group)
    return df.drop(columns=DIAG_COLS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de nettoyage complet, dans l'ordre."""
    n0 = len(df)
    df = replace_missing_token(df)
    df = drop_leakage_encounters(df)
    n_leak = n0 - len(df)
    # 3 lignes ont un genre « Unknown/Invalid » : on les écarte.
    df = df.loc[df["gender"] != "Unknown/Invalid"].copy()
    df = add_binary_target(df)
    df = group_icd9(df)
    if "weight" in df.columns:  # ~97 % manquant : inexploitable
        df = df.drop(columns=["weight"])
    df = df.reset_index(drop=True)
    log.info(
        "clean.done",
        rows_in=n0,
        rows_out=len(df),
        leakage_removed=n_leak,
        positive_rate=round(float(df["readmitted_30d"].mean()), 4),
    )
    return df


def main() -> None:
    """CLI : charge le brut, nettoie, sauvegarde le Parquet nettoyé."""
    from readmission_risk.common.config import settings
    from readmission_risk.data.loaders import load_raw

    df = clean(load_raw())
    out = settings.data_dir / settings.clean_filename
    df.to_parquet(out, index=False)
    log.info("clean.saved", path=str(out), shape=list(df.shape))


if __name__ == "__main__":
    main()

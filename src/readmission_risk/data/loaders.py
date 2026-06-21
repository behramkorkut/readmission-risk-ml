"""Chargement des données depuis le disque."""

from __future__ import annotations

import pandas as pd

from readmission_risk.common.config import settings


def load_raw() -> pd.DataFrame:
    """Charge la table brute (Parquet) produite par l'ingestion."""
    path = settings.data_dir / settings.raw_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Données absentes : {path}. Lance d'abord l'ingestion (readmission-ingest)."
        )
    return pd.read_parquet(path)


def load_clean() -> pd.DataFrame:
    """Charge la table nettoyée (Parquet) produite par l'étape de nettoyage."""
    path = settings.data_dir / settings.clean_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Données nettoyées absentes : {path}. Lance d'abord le nettoyage (readmission-clean)."
        )
    return pd.read_parquet(path)

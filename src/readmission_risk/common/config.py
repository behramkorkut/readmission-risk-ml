"""Configuration centralisée du projet (pydantic-settings).

Tout passe par une instance unique `settings` : chemins, graine aléatoire,
définition de la cible. Avantage : un seul endroit pour la config, typé et
surchargé par variables d'environnement / .env, et une reproductibilité
garantie par une graine fixe partagée partout.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet (…/readmission-risk-ml), calculée depuis ce fichier.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Reproductibilité ---
    random_seed: int = 42

    # --- Chemins (relatifs à la racine du projet) ---
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- Données / cible ---
    # Cible métier : réadmission à moins de 30 jours (classe positive).
    raw_filename: str = "diabetic_data.parquet"
    clean_filename: str = "diabetic_clean.parquet"
    target_col: str = "readmitted_30d"
    patient_id_col: str = "patient_nbr"  # clé de regroupement anti-fuite

    # --- Split ---
    test_size: float = 0.2  # part du jeu de test (hold-out), patient-disjoint

    # --- MLflow (utilisé à partir de l'étape 5) ---
    # Backend SQLite : le backend « fichier » (./mlruns) est déprécié en MLflow 3.x.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "readmission-30d"


settings = Settings()

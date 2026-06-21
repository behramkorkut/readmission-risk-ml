"""Prédiction de réadmission hospitalière à 30 jours (patients diabétiques).

Sous-paquets :
- data        : ingestion et chargement des données
- validation  : schéma Pandera et contrôles qualité
- features    : pipeline de préparation (sklearn)
- modeling    : entraînement, tuning, modèles
- evaluation  : métriques, calibration, conformal, SHAP, équité
- serving     : API de scoring (FastAPI)
- monitoring  : détection de drift (Evidently) et injection de défauts
- common      : configuration et utilitaires partagés
"""

__version__ = "0.1.0"

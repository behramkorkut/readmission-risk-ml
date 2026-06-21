"""Pipeline de préparation des features (scikit-learn).

Principe anti-fuite #3 : ce préprocesseur est un TRANSFORMER. On l'enchaîne avec
le modèle dans un Pipeline, qui sera ajusté À L'INTÉRIEUR de chaque fold de
validation croisée. Ainsi les statistiques d'imputation/scaling et les catégories
one-hot sont apprises uniquement sur le train du fold — jamais sur le test.

Choix dictés par la donnée :
- Numérique (aucun manquant ici) : imputation médiane (filet de sécurité) + scaling.
- Catégoriel : on traite l'absence comme une catégorie « Missing » (la non-mesure
  est cliniquement informative), puis one-hot avec regroupement des modalités rares.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Colonnes réellement numériques (comptes / durée). Les *_id sont des CODES,
# donc traités comme catégoriels, pas comme des quantités.
NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Colonnes à ne jamais utiliser comme features.
ID_COLS = ["encounter_id", "patient_nbr"]


def build_feature_spec(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str], list[str]]:
    """Détermine, à partir du TRAIN, les colonnes numériques / catégorielles.

    Écarte les identifiants, la cible, et les colonnes constantes (un seul niveau
    = aucun signal, ex. médicaments jamais prescrits comme 'examide').
    """
    exclude = set(ID_COLS) | {target_col}
    cols = [c for c in df.columns if c not in exclude]
    constant = [c for c in cols if df[c].nunique(dropna=False) <= 1]
    cols = [c for c in cols if c not in constant]
    numeric = [c for c in cols if c in NUMERIC_COLS]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical, constant


def _to_object(X):
    """Convertit en objets Python en préservant les NaN (les *_id entiers
    deviennent des catégories, sans transformer NaN en chaîne 'nan')."""
    return pd.DataFrame(X).astype(object)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Construit le ColumnTransformer (numérique + catégoriel)."""
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            # uniformise les types (int des *_id + str), garde les NaN
            ("to_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            # handle_unknown='ignore' : une modalité absente du train n'explose pas en test
            # min_frequency=10 : regroupe les modalités rares (maîtrise la dimension)
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

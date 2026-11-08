"""Chargement, contrôle et préparation des données de transactions.

Contrat des données (docs/plan_de_test.md, § 4) :
- colonnes Time, V1..V28, Amount (+ Class pour l'entraînement), sans doublon ni colonne inconnue ;
- valeurs numériques, finies, sans valeur manquante ;
- Amount >= 0 ; Class dans {0, 1}.
Toute donnée qui ne respecte pas le contrat est rejetée par une ValueError explicite.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
SEED = 42


def validate_data(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Vérifie le schéma et le contenu ; renvoie une copie validée ou lève ValueError."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Un DataFrame pandas est attendu, reçu : {type(df).__name__}")
    if df.empty:
        raise ValueError("Données vides : aucune ligne ou aucune colonne")

    columns = list(df.columns)
    duplicated = sorted({c for c in columns if columns.count(c) > 1})
    if duplicated:
        raise ValueError(f"Colonnes dupliquées : {duplicated}")

    expected = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in expected if c not in columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    allowed = set(FEATURES) | {TARGET}
    unknown = [c for c in columns if c not in allowed]
    if unknown:
        raise ValueError(f"Colonnes inconnues : {unknown}")

    checked = list(expected)
    not_numeric = [c for c in checked if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    if not_numeric:
        raise ValueError(f"Colonnes non numériques : {not_numeric}")

    values = df[checked].to_numpy(dtype=float)
    if np.isnan(values).any():
        cols = [c for c in checked if df[c].isna().any()]
        raise ValueError(f"Valeurs manquantes dans : {cols}")
    if np.isinf(values).any():
        cols = [c for c in checked if np.isinf(df[c].to_numpy(dtype=float)).any()]
        raise ValueError(f"Valeurs infinies dans : {cols}")

    if (df["Amount"] < 0).any():
        raise ValueError("Montant (Amount) négatif")
    if TARGET in columns:
        invalid = sorted(set(df[TARGET].unique()) - {0, 1})
        if invalid:
            raise ValueError(f"Valeurs de cible invalides (attendu 0 ou 1) : {invalid}")

    return df.copy()


MAX_FILE_SIZE_MB = 500  # le jeu complet pèse environ 150 Mo


def load_data(path, max_size_mb: float = MAX_FILE_SIZE_MB) -> pd.DataFrame:
    """Charge un fichier CSV et valide son contenu (extension, taille, format, schéma)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Extension non autorisée : {path.suffix or '(aucune)'} (seul .csv est accepté)")
    size_mb = path.stat().st_size / 1e6
    if size_mb > max_size_mb:
        raise ValueError(f"Fichier trop volumineux : {size_mb:.1f} Mo (maximum {max_size_mb} Mo)")
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(f"Fichier illisible : {path} ({exc})") from exc
    return validate_data(df)


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les lignes strictement identiques ; renvoie le DataFrame et le nombre supprimé."""
    deduplicated = df.drop_duplicates().reset_index(drop=True)
    return deduplicated, len(df) - len(deduplicated)


def preprocess_data(df: pd.DataFrame, require_target: bool = True):
    """Valide les données et renvoie les variables (ordre du contrat, float64) et la cible.

    La normalisation est réalisée dans le Pipeline du modèle (app/training.py), afin
    d'être ajustée uniquement sur les données d'entraînement.
    """
    data = validate_data(df, require_target=require_target)
    X = data[FEATURES].astype("float64")
    y = data[TARGET].astype("int64") if require_target else None
    return X, y


def split_data(X, y, val_size: float = 0.2, test_size: float = 0.2, seed: int = SEED):
    """Sépare en entraînement / validation / test, de façon stratifiée sur la cible."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val, stratify=y_trainval, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

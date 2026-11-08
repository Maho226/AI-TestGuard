"""Données de test synthétiques respectant le contrat (indépendantes du dataset Kaggle)."""

import numpy as np
import pandas as pd
import pytest

from app.preprocessing import FEATURES, TARGET


def make_transactions(n: int = 400, fraud_rate: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"Time": np.sort(rng.uniform(0, 172_800, n))}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)
    data["Amount"] = rng.exponential(80, n).round(2)
    y = np.zeros(n, dtype=int)
    y[: max(2, int(n * fraud_rate))] = 1
    rng.shuffle(y)
    # signal artificiel : les fraudes sont décalées sur quelques composantes
    for col in ("V4", "V11", "V14"):
        data[col] = data[col] + np.where(y == 1, 3.0 if col != "V14" else -3.0, 0.0)
    data[TARGET] = y
    return pd.DataFrame(data)[FEATURES + [TARGET]]


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture(scope="session")
def trained():
    """Pipeline entraîné sur des données synthétiques, avec son seuil et sa référence."""
    from app.prediction import training_reference
    from app.preprocessing import preprocess_data, split_data
    from app.training import train_model

    X, y = preprocess_data(make_transactions(n=1200, fraud_rate=0.08, seed=1))
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
    model = train_model(X_tr, y_tr)
    return {
        "model": model,
        "X_train": X_tr,
        "X_val": X_va,
        "X_test": X_te,
        "y_train": y_tr,
        "y_val": y_va,
        "y_test": y_te,
        "reference": training_reference(X_tr),
    }

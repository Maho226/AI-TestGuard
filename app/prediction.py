"""Production des prédictions, contrôle des entrées et alertes.

Les journaux ne contiennent jamais de valeurs de transaction : seulement des comptages (T18).
"""

import logging

import numpy as np
import pandas as pd

from app.preprocessing import FEATURES, preprocess_data

logger = logging.getLogger("ai_testguard")

DRIFT_ALERT_SHARE = 0.05  # alerte si plus de 5 % des valeurs sortent de la plage d'entraînement


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Probabilité de fraude, après contrôle du contrat des données."""
    features, _ = preprocess_data(X, require_target=False)
    proba = model.predict_proba(features[FEATURES])[:, 1]
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise RuntimeError("Le modèle a produit des probabilités invalides")
    return proba


def predict_fraud(model, X: pd.DataFrame, threshold: float = 0.5, reference: dict | None = None):
    """Renvoie (classes prédites, probabilités). Classe 1 si probabilité >= seuil.

    `reference` (plages min/max de l'entraînement) active les alertes de montant élevé et de dérive.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Seuil invalide : {threshold}")
    proba = predict_proba(model, X)
    y_pred = (proba >= threshold).astype(int)
    if reference is not None:
        check_alerts(X, reference)
    logger.info("Prédictions : %d transactions, %d signalées", len(y_pred), int(y_pred.sum()))
    return y_pred, proba


def training_reference(X_train: pd.DataFrame) -> dict:
    """Plages observées à l'entraînement, utilisées pour les alertes."""
    return {"min": X_train[FEATURES].min().to_dict(), "max": X_train[FEATURES].max().to_dict()}


def out_of_range_share(X: pd.DataFrame, reference: dict) -> pd.Series:
    """Part des valeurs hors de la plage d'entraînement, par variable."""
    low = pd.Series(reference["min"])
    high = pd.Series(reference["max"])
    data = X[FEATURES]
    return ((data < low) | (data > high)).mean()


def check_alerts(X: pd.DataFrame, reference: dict) -> list[str]:
    """Produit les alertes (montant inhabituel, dérive) sans journaliser de valeurs brutes."""
    alerts = []
    n_high = int((X["Amount"] > reference["max"]["Amount"]).sum())
    if n_high:
        alerts.append(f"montant_eleve:{n_high}")
        logger.warning("Alerte : %d transaction(s) au montant supérieur au maximum d'entraînement", n_high)
    share = out_of_range_share(X, reference)
    drifted = sorted(share[share > DRIFT_ALERT_SHARE].index)
    if drifted:
        alerts.append(f"derive:{','.join(drifted)}")
        logger.warning("Alerte de dérive : %d variable(s) hors plage d'entraînement", len(drifted))
    return alerts

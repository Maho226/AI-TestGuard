"""Production des prédictions, contrôle des entrées et alertes.

Les journaux ne contiennent jamais de valeurs de transaction : seulement des comptages (T18).
"""

import logging
import os
import platform
import time

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


def environment_profile() -> dict:
    """Description du matériel pour le protocole de mesure."""
    import psutil

    return {
        "processeur": platform.processor() or platform.machine(),
        "coeurs_logiques": os.cpu_count(),
        "memoire_totale_go": round(psutil.virtual_memory().total / 1e9, 1),
        "systeme": platform.platform(),
        "python": platform.python_version(),
    }


def _percentiles(values_ms) -> dict:
    v = np.asarray(values_ms)
    return {
        "moyenne_ms": float(v.mean()),
        "p50_ms": float(np.percentile(v, 50)),
        "p95_ms": float(np.percentile(v, 95)),
        "p99_ms": float(np.percentile(v, 99)),
        "max_ms": float(v.max()),
    }


def measure_performance(
    model,
    X: pd.DataFrame,
    threshold: float,
    n_single: int = 1000,
    warmup: int = 50,
    batch_sizes=(100, 1000, 10000),
    n_batch_repeats: int = 5,
    seed: int = 42,
) -> dict:
    """Latence d'une prédiction isolée (modèle seul / avec contrôle des entrées), débit par lot, mémoire.

    Protocole : perf_counter, `warmup` appels ignorés, `n_single` prédictions isolées tirées au hasard,
    `n_batch_repeats` répétitions par taille de lot. Une requête API ajouterait le traitement HTTP
    et la (dé)sérialisation JSON, non mesurés ici.
    """
    import psutil

    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(X), n_single + warmup)
    model_only, with_contract = [], []
    for i, r in enumerate(rows):
        x = X.iloc[[r]]
        t0 = time.perf_counter()
        model.predict_proba(x[FEATURES])
        t1 = time.perf_counter()
        predict_fraud(model, x, threshold=threshold)
        t2 = time.perf_counter()
        if i >= warmup:
            model_only.append((t1 - t0) * 1000)
            with_contract.append((t2 - t1) * 1000)

    throughput = {}
    for size in batch_sizes:
        batch = X.iloc[rng.integers(0, len(X), size)]
        durations = []
        for _ in range(n_batch_repeats):
            t0 = time.perf_counter()
            predict_fraud(model, batch, threshold=threshold)
            durations.append(time.perf_counter() - t0)
        throughput[str(size)] = {
            "duree_mediane_s": float(np.median(durations)),
            "predictions_par_s": float(size / np.median(durations)),
        }

    import tracemalloc

    big = X.iloc[rng.integers(0, len(X), max(batch_sizes))]
    tracemalloc.start()
    predict_fraud(model, big, threshold=threshold)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "latence_prediction_isolee_modele_seul": _percentiles(model_only),
        "latence_prediction_isolee_avec_controle_entrees": _percentiles(with_contract),
        "debit_par_lot": throughput,
        "memoire_processus_mo": round(psutil.Process().memory_info().rss / 1e6, 1),
        "memoire_pic_inference_lot_mo": round(peak / 1e6, 1),
        "protocole": {
            "repetitions_isolees": n_single,
            "echauffement": warmup,
            "tailles_de_lot": list(batch_sizes),
            "repetitions_par_lot": n_batch_repeats,
            "horloge": "time.perf_counter",
            "materiel": environment_profile(),
        },
    }


# ---------------------------------------------------------------- surveillance (Partie X)
PSI_WARNING = 0.10  # dérive modérée : à surveiller
PSI_ALERT = 0.25  # dérive importante : alerte
# Time est une position dans la fenêtre de collecte : sa distribution change par construction, il n'est pas suivi
DRIFT_FEATURES = [c for c in FEATURES if c != "Time"]


def drift_reference(X_train: pd.DataFrame, n_bins: int = 10) -> dict:
    """Bornes de déciles et proportions de référence de chaque variable (données d'entraînement)."""
    reference = {}
    for col in DRIFT_FEATURES:
        edges = np.unique(np.quantile(X_train[col], np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts = np.histogram(X_train[col], bins=edges)[0]
        reference[col] = {"edges": edges.tolist(), "ref": (counts / counts.sum()).tolist()}
    return reference


def population_stability_index(X: pd.DataFrame, reference: dict, eps: float = 1e-4) -> pd.Series:
    """PSI par variable : somme de (p_obs − p_ref) × ln(p_obs / p_ref) sur les intervalles de référence."""
    values = {}
    for col, ref in reference.items():
        observed = np.histogram(X[col], bins=np.asarray(ref["edges"]))[0] / len(X)
        p_ref = np.clip(np.asarray(ref["ref"]), eps, None)
        p_obs = np.clip(observed, eps, None)
        values[col] = float(np.sum((p_obs - p_ref) * np.log(p_obs / p_ref)))
    return pd.Series(values).sort_values(ascending=False)


def monitoring_report(X: pd.DataFrame, y_pred, reference: dict, expected_flag_rate: float, tolerance: float = 2.0):
    """Indicateurs disponibles SANS labels : dérive (PSI) et taux de transactions signalées."""
    psi = population_stability_index(X, reference)
    flag_rate = float(np.mean(y_pred))
    ratio = flag_rate / expected_flag_rate if expected_flag_rate > 0 else float("inf")
    alerts = [f"derive_importante:{c}" for c in psi[psi > PSI_ALERT].index]
    warnings = [f"derive_moderee:{c}" for c in psi[(psi > PSI_WARNING) & (psi <= PSI_ALERT)].index]
    if not 1 / tolerance <= ratio <= tolerance:
        alerts.append(f"taux_signalement:{flag_rate:.4%} (attendu {expected_flag_rate:.4%})")
    for a in alerts:
        logger.warning("Alerte de surveillance : %s", a.split(":")[0])
    return {
        "psi_max": float(psi.max()),
        "psi": psi.round(4).to_dict(),
        "flag_rate": flag_rate,
        "flag_rate_ratio": ratio,
        "alerts": alerts,
        "warnings": warnings,
    }

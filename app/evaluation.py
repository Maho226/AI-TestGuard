"""Métriques, choix du seuil, intervalles de confiance et décision d'acceptation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Critères fixés dans docs/plan_de_test.md (§ 5) avant toute évaluation sur le test
ACCEPTANCE = {
    "recall": (">=", 0.75),
    "precision": (">=", 0.30),
    "fpr": ("<=", 0.001),
    "pr_auc": (">=", 0.70),
    "roc_auc": (">=", 0.95),
}
MIN_VALIDATION_RECALL = 0.80


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    """Calcule les métriques de classification et la matrice de confusion."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError("y_true, y_pred et y_proba doivent avoir la même longueur")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Les deux classes doivent être présentes pour évaluer le modèle")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def choose_threshold(y_val, proba_val, min_recall: float = MIN_VALIDATION_RECALL) -> float:
    """Seuil maximisant la précision sous contrainte rappel >= min_recall (données de validation)."""
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    # precision/recall ont un élément de plus que thresholds : on ignore le dernier point
    eligible = np.where(recall[:-1] >= min_recall)[0]
    if eligible.size == 0:
        raise ValueError(f"Aucun seuil n'atteint un rappel de {min_recall} sur la validation")
    best = eligible[np.argmax(precision[eligible])]
    return float(thresholds[best])


def bootstrap_ci(y_true, y_pred, metric: str = "recall", n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Intervalle de confiance bootstrap d'une métrique (rappel, précision ou FPR)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    funcs = {
        "recall": lambda t, p: recall_score(t, p, zero_division=0),
        "precision": lambda t, p: precision_score(t, p, zero_division=0),
        "fpr": lambda t, p: ((p == 1) & (t == 0)).sum() / max((t == 0).sum(), 1),
    }
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        values.append(funcs[metric](y_true[idx], y_pred[idx]))
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def check_acceptance(metrics: dict, criteria: dict = ACCEPTANCE) -> dict:
    """Compare les métriques aux critères ; renvoie le détail et la décision."""
    details = {}
    for name, (op, limit) in criteria.items():
        value = metrics[name]
        ok = value >= limit if op == ">=" else value <= limit
        details[name] = {"value": value, "criterion": f"{op} {limit}", "passed": bool(ok)}
    if not details["recall"]["passed"]:
        decision = "rejeté"
    elif all(d["passed"] for d in details.values()):
        decision = "accepté pour un pilote"
    else:
        decision = "à investiguer"
    return {"criteria": details, "decision": decision}


def perturb(X, columns, kind: str, amplitude: float, scale=None, seed: int = 42):
    """Renvoie une copie perturbée de X et le nombre de valeurs rendues invalides.

    - kind="gaussian" : bruit N(0, amplitude × écart-type d'entraînement) sur chaque colonne (V1–V28) ;
    - kind="relative" : variation multiplicative uniforme de ± amplitude (Amount).
    Les montants rendus négatifs sont comptés comme invalides puis ramenés à 0.
    """
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    for col in columns:
        if kind == "gaussian":
            sigma = amplitude * (scale[col] if scale is not None else X[col].std())
            Xp[col] = X[col] + rng.normal(0.0, sigma, len(X))
        elif kind == "relative":
            Xp[col] = X[col] * (1 + rng.uniform(-amplitude, amplitude, len(X)))
        else:
            raise ValueError(f"Perturbation inconnue : {kind}")
    n_invalid = int((Xp["Amount"] < 0).sum()) if "Amount" in Xp else 0
    if n_invalid:
        Xp["Amount"] = Xp["Amount"].clip(lower=0)
    return Xp, n_invalid


def sensitivity(model, X, y, threshold: float, X_perturbed, n_invalid: int = 0) -> dict:
    """Mesure l'effet d'une perturbation : probabilités, changements de classe, métriques."""
    p0 = model.predict_proba(X)[:, 1]
    p1 = model.predict_proba(X_perturbed)[:, 1]
    c0 = (p0 >= threshold).astype(int)
    c1 = (p1 >= threshold).astype(int)
    m0 = evaluate_model(y, c0, p0)
    m1 = evaluate_model(y, c1, p1)
    delta = np.abs(p1 - p0)
    return {
        "mean_abs_proba_change": float(delta.mean()),
        "p99_abs_proba_change": float(np.quantile(delta, 0.99)),
        "class_changes": int((c0 != c1).sum()),
        "class_changes_frauds": int(((c0 != c1) & (np.asarray(y) == 1)).sum()),
        "recall_before": m0["recall"],
        "recall_after": m1["recall"],
        "precision_before": m0["precision"],
        "precision_after": m1["precision"],
        "fpr_before": m0["fpr"],
        "fpr_after": m1["fpr"],
        "n_invalid": n_invalid,
    }


# Tolérances de non-régression (plan de test, T16) : dégradation maximale admise par rapport à la référence
REGRESSION_TOLERANCES = {"pr_auc": 0.02, "recall": 0.03, "precision": 0.05, "fpr": 0.0005}
HIGHER_IS_BETTER = {"pr_auc": True, "recall": True, "precision": True, "fpr": False}


def compare_to_baseline(new: dict, baseline: dict, tolerances: dict = REGRESSION_TOLERANCES) -> dict:
    """Compare les métriques d'une nouvelle version à la référence figée.

    - aucune dégradation au-delà de la tolérance : « accepter » ;
    - dégradation au-delà de la tolérance : « investiguer » (le contrôle automatisé échoue) ;
    - dégradation au-delà du double de la tolérance : « rejeter ».
    """
    details, worst = {}, 0.0
    for metric, tol in tolerances.items():
        ref, cur = baseline[metric], new[metric]
        degradation = (ref - cur) if HIGHER_IS_BETTER[metric] else (cur - ref)
        details[metric] = {
            "reference": ref,
            "nouvelle": cur,
            "ecart": cur - ref,
            "tolerance": tol,
            "regression": bool(degradation > tol),
        }
        worst = max(worst, degradation / tol)
    if worst > 2:
        decision = "rejeter"
    elif worst > 1:
        decision = "investiguer"
    else:
        decision = "accepter"
    return {"metrics": details, "decision": decision}

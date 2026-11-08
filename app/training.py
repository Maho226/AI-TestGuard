"""Construction, entraînement, réglage, sauvegarde et chargement du pipeline."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.preprocessing import FEATURES, SEED

# Grilles d'hyperparamètres explorées sur la validation (jamais sur le test)
PARAM_GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "decision_tree": [{"max_depth": d, "min_samples_leaf": 20} for d in (4, 6, 8)],
}


def build_pipeline(model_name: str = "logistic_regression", seed: int = SEED, **params) -> Pipeline:
    """Pipeline StandardScaler + classifieur ; le scaler n'est ajusté que lors du fit."""
    if model_name == "logistic_regression":
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed, **params)
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def train_model(X, y, model_name: str = "logistic_regression", seed: int = SEED, **params) -> Pipeline:
    """Entraîne le pipeline sur les données fournies (données d'entraînement uniquement)."""
    if len(X) != len(y):
        raise ValueError(f"Dimensions incohérentes : X={len(X)} lignes, y={len(y)} valeurs")
    if len(np.unique(y)) < 2:
        raise ValueError("Les données d'entraînement doivent contenir les deux classes")
    pipeline = build_pipeline(model_name, seed=seed, **params)
    pipeline.fit(X[FEATURES], y)
    return pipeline


def tune_on_validation(X_train, y_train, X_val, y_val, model_name: str, seed: int = SEED):
    """Choisit les hyperparamètres maximisant la PR-AUC sur la validation."""
    results = []
    for params in PARAM_GRIDS[model_name]:
        pipeline = train_model(X_train, y_train, model_name, seed=seed, **params)
        score = average_precision_score(y_val, pipeline.predict_proba(X_val[FEATURES])[:, 1])
        results.append({"params": params, "val_pr_auc": float(score), "pipeline": pipeline})
    best = max(results, key=lambda r: r["val_pr_auc"])
    return best["pipeline"], best["params"], [{k: r[k] for k in ("params", "val_pr_auc")} for r in results]


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def save_model(pipeline: Pipeline, path, threshold: float, metadata: dict | None = None) -> Path:
    """Sauvegarde le pipeline, le seuil de décision et les métadonnées."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Seuil invalide : {threshold}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "features": FEATURES,
        "metadata": {
            **(metadata or {}),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": environment_info(),
        },
    }
    joblib.dump(bundle, path)
    path.with_suffix(".json").write_text(json.dumps(bundle["metadata"] | {"threshold": threshold}, indent=2))
    return path


def load_model(path) -> dict:
    """Charge un modèle sauvegardé et vérifie sa structure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # fichier corrompu ou d'un autre format
        raise ValueError(f"Modèle illisible : {path} ({exc})") from exc
    if not isinstance(bundle, dict) or not {"pipeline", "threshold", "features"} <= bundle.keys():
        raise ValueError(f"Contenu de modèle invalide : {path}")
    if list(bundle["features"]) != FEATURES:
        raise ValueError("Le modèle a été entraîné avec un schéma de variables différent")
    return bundle


def run_training_pipeline(data_path, model_path, model_name: str = "logistic_regression", seed: int = SEED) -> dict:
    """Scénario complet : chargement, contrôle, séparation, réglage et seuil sur la validation,
    sauvegarde, rechargement, prédiction sur le test et rapport de métriques."""
    from app.evaluation import choose_threshold, evaluate_model
    from app.prediction import predict_fraud
    from app.preprocessing import load_data, preprocess_data, remove_duplicates, split_data

    raw = load_data(data_path)
    data, n_duplicates = remove_duplicates(raw)
    X, y = preprocess_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=seed)

    pipeline, params, _ = tune_on_validation(X_train, y_train, X_val, y_val, model_name, seed=seed)
    proba_val = pipeline.predict_proba(X_val[FEATURES])[:, 1]
    threshold = choose_threshold(y_val, proba_val)

    save_model(pipeline, model_path, threshold, {"model": model_name, "params": params, "seed": seed})
    bundle = load_model(model_path)
    y_pred, proba_test = predict_fraud(bundle["pipeline"], X_test, threshold=bundle["threshold"])

    return {
        "n_raw": len(raw),
        "n_duplicates": n_duplicates,
        "params": params,
        "threshold": threshold,
        "validation": evaluate_model(y_val, (proba_val >= threshold).astype(int), proba_val),
        "test": evaluate_model(y_test, y_pred, proba_test),
        "pipeline": pipeline,
        "bundle": bundle,
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        "proba_test": proba_test,
    }

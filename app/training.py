"""Construction, entraînement, réglage, sauvegarde et chargement du pipeline.

Les versions du modèle sont décrites dans MODEL_VERSIONS. La version déployée est CURRENT_VERSION ;
elle est comparée à la référence figée (reports/baseline_*.json) par les tests de non-régression.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.preprocessing import FEATURES, SEED

# Grilles d'hyperparamètres explorées sur la validation (jamais sur le test)
PARAM_GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "decision_tree": [{"max_depth": d, "min_samples_leaf": 20} for d in (4, 6, 8)],
}

MODEL_VERSIONS = {
    # v1 : référence (Partie III) — rejetée, anomalies A1 (saturation) et A3 (sensibilité au montant)
    "v1": {"model_name": "logistic_regression", "log_amount": False, "grid": PARAM_GRIDS["logistic_regression"]},
    # v2 : log(1 + Amount) contre A3 ; pondération des fraudes modérée contre A1
    "v2": {
        "model_name": "logistic_regression",
        "log_amount": True,
        "grid": [{"C": c, "class_weight": {0: 1, 1: w}} for c in (0.1, 1.0) for w in (1, 10, 50)],
    },
}
CURRENT_VERSION = "v2"


def log_amount(X: pd.DataFrame) -> pd.DataFrame:
    """Remplace Amount par log(1 + Amount) : atténue l'effet d'échelle des montants."""
    X = X.copy()
    X["Amount"] = np.log1p(X["Amount"])
    return X


def build_pipeline(
    model_name: str = "logistic_regression", seed: int = SEED, log_amount_feature: bool = False, **params
) -> Pipeline:
    """Pipeline [log(Amount)] + StandardScaler + classifieur ; les transformations ne sont ajustées qu'au fit."""
    params.setdefault("class_weight", "balanced")
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000, random_state=seed, **params)
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
    steps = [("log_amount", FunctionTransformer(log_amount))] if log_amount_feature else []
    return Pipeline(steps + [("scaler", StandardScaler()), ("model", model)])


def train_model(X, y, model_name: str = "logistic_regression", seed: int = SEED, **params) -> Pipeline:
    """Entraîne le pipeline sur les données fournies (données d'entraînement uniquement)."""
    if len(X) != len(y):
        raise ValueError(f"Dimensions incohérentes : X={len(X)} lignes, y={len(y)} valeurs")
    if len(np.unique(y)) < 2:
        raise ValueError("Les données d'entraînement doivent contenir les deux classes")
    pipeline = build_pipeline(model_name, seed=seed, **params)
    pipeline.fit(X[FEATURES], y)
    return pipeline


def tune_on_validation(
    X_train, y_train, X_val, y_val, model_name: str, seed: int = SEED, grid=None, log_amount_feature: bool = False
):
    """Choisit les hyperparamètres maximisant la PR-AUC sur la validation."""
    results = []
    for params in grid or PARAM_GRIDS[model_name]:
        pipeline = train_model(
            X_train, y_train, model_name, seed=seed, log_amount_feature=log_amount_feature, **dict(params)
        )
        score = average_precision_score(y_val, pipeline.predict_proba(X_val[FEATURES])[:, 1])
        results.append({"params": params, "val_pr_auc": float(score), "pipeline": pipeline})
    best = max(results, key=lambda r: r["val_pr_auc"])
    return best["pipeline"], best["params"], [{k: r[k] for k in ("params", "val_pr_auc")} for r in results]


def tune_version(X_train, y_train, X_val, y_val, version: str = CURRENT_VERSION, seed: int = SEED):
    """Réglage d'une version déclarée dans MODEL_VERSIONS."""
    if version not in MODEL_VERSIONS:
        raise ValueError(f"Version inconnue : {version}")
    cfg = MODEL_VERSIONS[version]
    return tune_on_validation(
        X_train,
        y_train,
        X_val,
        y_val,
        cfg["model_name"],
        seed=seed,
        grid=cfg["grid"],
        log_amount_feature=cfg["log_amount"],
    )


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def code_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnue"


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    sidecar = bundle["metadata"] | {"threshold": threshold, "sha256": file_sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str))
    return path


def load_model(path, expected_sha256: str | None = None) -> dict:
    """Charge un modèle sauvegardé et vérifie sa structure.

    joblib/pickle peut exécuter du code à la lecture : on ne charge que des fichiers de confiance,
    et l'empreinte SHA-256 enregistrée à la sauvegarde permet de détecter un fichier modifié.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"Intégrité du modèle non vérifiée (empreinte SHA-256 différente) : {path}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # fichier corrompu ou d'un autre format
        raise ValueError(f"Modèle illisible : {path} ({exc})") from exc
    if not isinstance(bundle, dict) or not {"pipeline", "threshold", "features"} <= bundle.keys():
        raise ValueError(f"Contenu de modèle invalide : {path}")
    if list(bundle["features"]) != FEATURES:
        raise ValueError("Le modèle a été entraîné avec un schéma de variables différent")
    return bundle


def _prepare(data_path, seed: int):
    from app.preprocessing import load_data, preprocess_data, remove_duplicates, split_data

    raw = load_data(data_path)
    data, n_duplicates = remove_duplicates(raw)
    X, y = preprocess_data(data)
    return raw, n_duplicates, y, split_data(X, y, seed=seed)


def run_training_pipeline(data_path, model_path, version: str = CURRENT_VERSION, seed: int = SEED) -> dict:
    """Scénario complet : chargement, contrôle, séparation, réglage et seuil sur la validation,
    sauvegarde, rechargement, prédiction sur le test et rapport de métriques."""
    from app.evaluation import choose_threshold, evaluate_model
    from app.prediction import predict_fraud

    raw, n_duplicates, _, (X_train, X_val, X_test, y_train, y_val, y_test) = _prepare(data_path, seed)
    pipeline, params, _ = tune_version(X_train, y_train, X_val, y_val, version, seed=seed)
    proba_val = pipeline.predict_proba(X_val[FEATURES])[:, 1]
    threshold = choose_threshold(y_val, proba_val)

    save_model(pipeline, model_path, threshold, {"version": version, "params": params, "seed": seed})
    bundle = load_model(model_path)
    y_pred, proba_test = predict_fraud(bundle["pipeline"], X_test, threshold=bundle["threshold"])

    return {
        "version": version,
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


def compute_reference(data_path, version: str = CURRENT_VERSION, seed: int = SEED) -> dict:
    """Mesures de référence d'une version, sur le jeu de VALIDATION fixe (le test n'est pas utilisé)."""
    from app.evaluation import choose_threshold, evaluate_model

    raw, n_duplicates, y, (X_train, X_val, _, y_train, y_val, _) = _prepare(data_path, seed)
    pipeline, params, _ = tune_version(X_train, y_train, X_val, y_val, version, seed=seed)
    proba_val = pipeline.predict_proba(X_val[FEATURES])[:, 1]
    threshold = choose_threshold(y_val, proba_val)
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": code_version(),
        "model": MODEL_VERSIONS[version]["model_name"],
        "params": {k: (str(v) if isinstance(v, dict) else v) for k, v in params.items()},
        "log_amount": MODEL_VERSIONS[version]["log_amount"],
        "seed": seed,
        "environment": environment_info(),
        "dataset": {
            "file": Path(data_path).name,
            "sha256": file_sha256(data_path),
            "rows": len(raw),
            "duplicates_removed": n_duplicates,
            "fraud_rate": float(y.mean()),
            "validation_rows": len(y_val),
            "validation_frauds": int(y_val.sum()),
        },
        "threshold": threshold,
        "validation_metrics": evaluate_model(y_val, (proba_val >= threshold).astype(int), proba_val),
        "saturated_probabilities": int((proba_val == 1.0).sum()),
    }


def main(argv=None):
    """Création MANUELLE d'une référence : python -m app.training --data ... --version v1 --output ..."""
    parser = argparse.ArgumentParser(description="Crée un fichier de référence (baseline) figé.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--version", default=CURRENT_VERSION)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true", help="écraser une référence existante")
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists() and not args.force:
        raise SystemExit(f"La référence {out} existe déjà : elle n'est jamais recalculée automatiquement (--force).")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compute_reference(args.data, args.version), indent=2, ensure_ascii=False))
    print(f"Référence écrite : {out}")


if __name__ == "__main__":
    main()

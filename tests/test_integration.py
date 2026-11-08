"""Tests d'intégration du pipeline de bout en bout (Partie IV).

Scénario exécuté sur l'échantillon versionné : chargement → schéma → séparation → réglage et seuil
sur la validation → sauvegarde → rechargement → prédiction sur le test → métriques.
Exigences couvertes : T03, T05, T10, T11, T12, T14 (docs/plan_de_test.md).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.evaluation import choose_threshold
from app.prediction import predict_fraud
from app.preprocessing import FEATURES, TARGET
from app.training import load_model, run_training_pipeline, save_model

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "creditcard_sample.csv"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    return run_training_pipeline(SAMPLE, model_path) | {"model_path": model_path}


# ---------------------------------------------------------------- scénario complet
def test_pipeline_runs_end_to_end(run):
    assert run["model_path"].is_file()
    assert run["model_path"].with_suffix(".json").is_file()
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr"):
        assert 0.0 <= run["test"][key] <= 1.0
    assert run["test"]["tp"] + run["test"]["fn"] == int(run["splits"]["y_test"].sum())


# ---------------------------------------------------------------- T03 / dimensions
def test_t03_no_non_finite_values_after_preprocessing(run):
    for name in ("X_train", "X_val", "X_test"):
        assert np.isfinite(run["splits"][name].to_numpy()).all()


def test_dimensions_are_consistent(run):
    s = run["splits"]
    assert len(s["X_train"]) + len(s["X_val"]) + len(s["X_test"]) == run["n_raw"] - run["n_duplicates"]
    for X, y in ((s["X_train"], s["y_train"]), (s["X_val"], s["y_val"]), (s["X_test"], s["y_test"])):
        assert X.shape[1] == 30 and len(X) == len(y)
    assert run["proba_test"].shape == (len(s["X_test"]),)


# ---------------------------------------------------------------- T10 fuites de données
def test_t10_scaler_fitted_on_training_data_only(run):
    scaler = run["pipeline"].named_steps["scaler"]
    X_train = run["splits"]["X_train"]
    assert scaler.n_samples_seen_ == len(X_train)
    np.testing.assert_allclose(scaler.mean_, X_train[FEATURES].mean().to_numpy(), rtol=1e-10)
    all_data = pd.concat([X_train, run["splits"]["X_val"], run["splits"]["X_test"]])
    assert not np.allclose(scaler.mean_, all_data[FEATURES].mean().to_numpy(), rtol=1e-6)


def test_t10_no_shared_transactions_between_train_and_test(run):
    s = run["splits"]
    assert set(s["X_train"].index).isdisjoint(s["X_test"].index)
    assert pd.merge(s["X_train"], s["X_test"], how="inner").empty


def test_t10_duplicates_cannot_leak(tmp_path):
    """Des transactions dupliquées dans le fichier source ne doivent pas se retrouver des deux côtés."""
    df = pd.read_csv(SAMPLE)
    noisy = pd.concat([df, df.iloc[:300]], ignore_index=True)
    path = tmp_path / "avec_doublons.csv"
    noisy.to_csv(path, index=False)
    result = run_training_pipeline(path, tmp_path / "m.joblib")
    assert result["n_duplicates"] >= 300
    assert pd.merge(result["splits"]["X_train"], result["splits"]["X_test"], how="inner").empty


# ---------------------------------------------------------------- T11 / T05 / T12
def test_t11_stratification_preserved(run):
    s = run["splits"]
    overall = pd.concat([s["y_train"], s["y_val"], s["y_test"]]).mean()
    for part in ("y_train", "y_val", "y_test"):
        assert abs(s[part].mean() - overall) <= 0.02


def test_t05_t12_threshold_comes_from_validation_only(run):
    s = run["splits"]
    proba_val = run["pipeline"].predict_proba(s["X_val"][FEATURES])[:, 1]
    assert run["threshold"] == choose_threshold(s["y_val"], proba_val)
    assert run["validation"]["recall"] >= 0.80
    assert set(s["X_val"].index).isdisjoint(s["X_test"].index)


# ---------------------------------------------------------------- T14 sauvegarde / rechargement
def test_t14_reloaded_pipeline_gives_identical_predictions(run):
    bundle = load_model(run["model_path"])
    X_test = run["splits"]["X_test"]
    before = run["pipeline"].predict_proba(X_test[FEATURES])[:, 1]
    after = bundle["pipeline"].predict_proba(X_test[FEATURES])[:, 1]
    assert np.max(np.abs(before - after)) <= 1e-9
    y_before = (before >= run["threshold"]).astype(int)
    y_after, _ = predict_fraud(bundle["pipeline"], X_test, threshold=bundle["threshold"])
    np.testing.assert_array_equal(y_before, y_after)


def test_t14_saved_metadata_is_complete(run):
    bundle = load_model(run["model_path"])
    assert bundle["threshold"] == pytest.approx(run["threshold"])
    assert bundle["features"] == FEATURES
    env = bundle["metadata"]["environment"]
    assert {"python", "scikit-learn", "numpy", "joblib"} <= env.keys()
    assert bundle["metadata"]["seed"] == 42


# ---------------------------------------------------------------- erreurs de chargement et d'entrée
def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_model(tmp_path / "absent.joblib")


def test_load_model_corrupted_file(tmp_path):
    bad = tmp_path / "corrompu.joblib"
    bad.write_bytes(b"ceci n'est pas un modele")
    with pytest.raises(ValueError, match="illisible"):
        load_model(bad)


def test_load_model_invalid_content(tmp_path):
    path = tmp_path / "mauvais.joblib"
    joblib.dump({"pipeline": None}, path)
    with pytest.raises(ValueError, match="invalide"):
        load_model(path)


def test_load_model_wrong_schema(run, tmp_path):
    bundle = load_model(run["model_path"])
    bundle["features"] = FEATURES[:-1]
    path = tmp_path / "schema.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="schéma"):
        load_model(path)


def test_save_model_rejects_invalid_threshold(run, tmp_path):
    with pytest.raises(ValueError, match="Seuil invalide"):
        save_model(run["pipeline"], tmp_path / "m.joblib", threshold=1.5)


def test_pipeline_rejects_invalid_input_file(tmp_path):
    df = pd.read_csv(SAMPLE).drop(columns=TARGET)
    path = tmp_path / "sans_cible.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        run_training_pipeline(path, tmp_path / "m.joblib")

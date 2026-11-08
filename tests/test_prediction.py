"""Tests de la prédiction, de l'évaluation et du choix du seuil (Parties III et VIII).

Exigences couvertes : T04, T09, T12, T13 (logique de décision), T18 (docs/plan_de_test.md).
"""

import logging
import re

import numpy as np
import pytest

from app.evaluation import (
    ACCEPTANCE,
    bootstrap_ci,
    check_acceptance,
    choose_threshold,
    evaluate_model,
)
from app.prediction import check_alerts, predict_fraud, predict_proba
from app.training import build_pipeline, train_model


# ---------------------------------------------------------------- T04 contrat de sortie
def test_t04_probabilities_are_valid(trained):
    proba = predict_proba(trained["model"], trained["X_test"])
    assert proba.shape == (len(trained["X_test"]),)
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_t04_classes_follow_threshold(trained):
    y_pred, proba = predict_fraud(trained["model"], trained["X_test"], threshold=0.7)
    assert set(np.unique(y_pred)) <= {0, 1}
    np.testing.assert_array_equal(y_pred, (proba >= 0.7).astype(int))


@pytest.mark.parametrize("threshold, expected", [(0.0, 1), (1.0 + 1e-12, None)])
def test_t04_threshold_limits(trained, threshold, expected):
    if expected is None:
        with pytest.raises(ValueError, match="Seuil invalide"):
            predict_fraud(trained["model"], trained["X_test"], threshold=threshold)
    else:
        y_pred, _ = predict_fraud(trained["model"], trained["X_test"], threshold=threshold)
        assert (y_pred == expected).all()


def test_t04_single_transaction(trained):
    y_pred, proba = predict_fraud(trained["model"], trained["X_test"].iloc[[0]])
    assert y_pred.shape == (1,) and proba.shape == (1,)


def test_t04_model_detects_synthetic_signal(trained):
    """Sur des données au signal net, le modèle doit séparer les classes (sinon le pipeline est cassé)."""
    proba = predict_proba(trained["model"], trained["X_test"])
    assert evaluate_model(trained["y_test"], (proba >= 0.5).astype(int), proba)["roc_auc"] > 0.95


# ---------------------------------------------------------------- T09 entrées invalides
def test_t09_prediction_rejects_missing_column(trained):
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        predict_proba(trained["model"], trained["X_test"].drop(columns="V10"))


def test_t09_prediction_rejects_nan(trained):
    X = trained["X_test"].copy()
    X.iloc[0, 3] = np.nan
    with pytest.raises(ValueError, match="Valeurs manquantes"):
        predict_proba(trained["model"], X)


def test_t09_training_rejects_single_class(trained):
    with pytest.raises(ValueError, match="deux classes"):
        train_model(trained["X_train"], np.zeros(len(trained["X_train"]), dtype=int))


def test_t09_training_rejects_inconsistent_dimensions(trained):
    with pytest.raises(ValueError, match="Dimensions incohérentes"):
        train_model(trained["X_train"], trained["y_train"].iloc[:-1])


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Modèle inconnu"):
        build_pipeline("random_forest")


# ---------------------------------------------------------------- T18 sécurité des entrées et journaux
def test_t18_extra_field_is_rejected(trained):
    with pytest.raises(ValueError, match="Colonnes inconnues"):
        predict_proba(trained["model"], trained["X_test"].assign(customer_name="Jean Dupont"))


def test_t18_logs_contain_no_transaction_values(trained, caplog):
    X = trained["X_test"].copy()
    X.iloc[0, X.columns.get_loc("Amount")] = 987654.32
    with caplog.at_level(logging.INFO, logger="ai_testguard"):
        predict_fraud(trained["model"], X, reference=trained["reference"])
    text = caplog.text
    assert "987654" not in text
    # aucun nombre décimal (valeur de variable) dans les messages : seulement des comptages
    assert not re.search(r"\d+\.\d+", " ".join(r.getMessage() for r in caplog.records))
    assert any("Alerte" in r.getMessage() for r in caplog.records)


def test_t18_high_amount_triggers_alert(trained):
    X = trained["X_test"].iloc[:5].copy()
    X["Amount"] = trained["reference"]["max"]["Amount"] * 10
    assert any(a.startswith("montant_eleve:5") for a in check_alerts(X, trained["reference"]))


# ---------------------------------------------------------------- métriques (valeurs calculées à la main)
def test_evaluate_model_known_values():
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.1, 0.3, 0.6, 0.7, 0.9, 0.8, 0.65, 0.4])
    m = evaluate_model(y_true, y_pred, y_proba)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (3, 4, 2, 1)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["fpr"] == pytest.approx(2 / 6)
    assert m["fnr"] == pytest.approx(1 / 4)
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))


def test_accuracy_is_misleading_on_imbalanced_data():
    """Un modèle qui ne détecte aucune fraude affiche 99,8 % d'accuracy et 0 de rappel."""
    y_true = np.array([1] * 2 + [0] * 998)
    y_pred = np.zeros(1000, dtype=int)
    m = evaluate_model(y_true, y_pred, np.full(1000, 0.01))
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0.0 and m["tp"] == 0


def test_evaluate_model_rejects_single_class():
    with pytest.raises(ValueError, match="deux classes"):
        evaluate_model([0, 0], [0, 0], [0.1, 0.2])


# ---------------------------------------------------------------- T12 seuil choisi sur la validation
def test_t12_threshold_meets_recall_constraint():
    y = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([np.linspace(0.0, 0.6, 90), np.linspace(0.3, 0.95, 10)])
    threshold = choose_threshold(y, proba, min_recall=0.8)
    recall = ((proba >= threshold) & (y == 1)).sum() / 10
    assert recall >= 0.8


def test_t12_threshold_maximizes_precision_among_eligible():
    y = np.array([0, 0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.5, 0.6, 0.8, 0.9])
    # seuils possibles avec rappel >= 2/3 : 0.5 (précision 3/4) ou 0.8 (précision 2/2)
    assert choose_threshold(y, proba, min_recall=0.66) == pytest.approx(0.8)


def test_t12_threshold_impossible_constraint_is_explicit():
    with pytest.raises(ValueError, match="Aucun seuil"):
        choose_threshold(np.array([0, 1]), np.array([0.2, 0.8]), min_recall=1.1)


def test_t12_threshold_on_trained_model(trained):
    proba_val = predict_proba(trained["model"], trained["X_val"])
    threshold = choose_threshold(trained["y_val"], proba_val)
    assert 0.0 <= threshold <= 1.0
    recall_val = ((proba_val >= threshold) & (trained["y_val"] == 1)).sum() / trained["y_val"].sum()
    assert recall_val >= 0.80


# ---------------------------------------------------------------- intervalles et décision
def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_pred = np.where(rng.random(500) < 0.8, y_true, 1 - y_true)
    low, high = bootstrap_ci(y_true, y_pred, "recall", n_boot=300)
    point = ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
    assert low <= point <= high and high - low < 0.2


@pytest.mark.parametrize(
    "overrides, decision",
    [
        ({}, "accepté pour un pilote"),
        ({"precision": 0.25}, "à investiguer"),
        ({"recall": 0.70}, "rejeté"),
    ],
)
def test_t13_acceptance_decision(overrides, decision):
    metrics = {"recall": 0.85, "precision": 0.6, "fpr": 0.0005, "pr_auc": 0.8, "roc_auc": 0.97} | overrides
    result = check_acceptance(metrics)
    assert result["decision"] == decision
    assert set(result["criteria"]) == set(ACCEPTANCE)

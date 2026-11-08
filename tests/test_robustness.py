"""Tests de robustesse et de sensibilité (Partie V).

Pour chaque scénario du plan de test (§ 6), le comportement attendu est : rejet, traitement ou alerte.
Les perturbations mesurent une sensibilité ; elles n'imposent pas une invariance universelle.
Exigences couvertes : T04, T09, T15 (docs/plan_de_test.md).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.evaluation import perturb, sensitivity
from app.prediction import check_alerts, predict_fraud, training_reference
from app.preprocessing import FEATURES
from app.training import run_training_pipeline

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "creditcard_sample.csv"
V_COLS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    run = run_training_pipeline(SAMPLE, tmp_path_factory.mktemp("m") / "model.joblib")
    s = run["splits"]
    return {
        "model": run["pipeline"],
        "threshold": run["threshold"],
        "X": s["X_val"][FEATURES],
        "y": s["y_val"],
        "std": s["X_train"][FEATURES].std(),
        "reference": training_reference(s["X_train"]),
    }


# ---------------------------------------------------------------- scénarios rejetés
@pytest.mark.parametrize(
    "scenario, mutate, message",
    [
        ("valeur manquante", lambda X: X.assign(V3=np.where(np.arange(len(X)) == 0, np.nan, X["V3"])), "manquantes"),
        ("plusieurs variables manquantes", lambda X: X.assign(V1=np.nan, Amount=np.nan), "manquantes"),
        ("valeur infinie", lambda X: X.assign(V5=np.where(np.arange(len(X)) == 0, np.inf, X["V5"])), "infinies"),
        ("valeur non numérique", lambda X: X.assign(Amount="cent euros"), "non numériques"),
        ("schéma invalide : colonne absente", lambda X: X.drop(columns="V28"), "manquantes"),
        ("schéma invalide : colonne inconnue", lambda X: X.assign(merchant_id=1), "inconnues"),
    ],
)
def test_t15_invalid_inputs_are_rejected(ctx, scenario, mutate, message):
    with pytest.raises(ValueError, match=message):
        predict_fraud(ctx["model"], mutate(ctx["X"].copy()), threshold=ctx["threshold"])


# ---------------------------------------------------------------- scénarios traités avec alerte
@pytest.mark.parametrize("factor", [10, 1_000, 1_000_000])
def test_t15_extreme_amount_is_processed_with_alert(ctx, factor):
    X = ctx["X"].iloc[:20].copy()
    X["Amount"] = ctx["reference"]["max"]["Amount"] * factor
    y_pred, proba = predict_fraud(ctx["model"], X, threshold=ctx["threshold"])
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()
    assert any(a.startswith("montant_eleve:20") for a in check_alerts(X, ctx["reference"]))


def test_t15_distribution_shift_triggers_drift_alert(ctx):
    shifted = ctx["X"].copy()
    shifted["V14"] = shifted["V14"] + 10 * ctx["std"]["V14"]
    alerts = check_alerts(shifted, ctx["reference"])
    assert any(a.startswith("derive:") and "V14" in a for a in alerts)
    _, proba = predict_fraud(ctx["model"], shifted, threshold=ctx["threshold"])
    assert np.isfinite(proba).all()


def test_t15_in_distribution_data_raises_no_drift_alert(ctx):
    assert not any(a.startswith("derive:") for a in check_alerts(ctx["X"], ctx["reference"]))


# ---------------------------------------------------------------- sensibilité aux perturbations limitées
def test_t15_small_perturbation_has_limited_effect(ctx):
    Xp, n_invalid = perturb(ctx["X"], V_COLS, "gaussian", 0.01, scale=ctx["std"])
    s = sensitivity(ctx["model"], ctx["X"], ctx["y"], ctx["threshold"], Xp, n_invalid)
    assert s["mean_abs_proba_change"] < 0.02
    assert s["class_changes"] <= 0.02 * len(ctx["X"])
    assert abs(s["recall_after"] - s["recall_before"]) <= 0.05


def test_t15_sensitivity_grows_with_amplitude(ctx):
    changes = []
    for amplitude in (0.01, 0.05, 0.2):
        Xp, _ = perturb(ctx["X"], V_COLS, "gaussian", amplitude, scale=ctx["std"])
        changes.append(sensitivity(ctx["model"], ctx["X"], ctx["y"], ctx["threshold"], Xp)["mean_abs_proba_change"])
    assert changes[0] <= changes[1] <= changes[2]


def test_t15_amount_perturbation_counts_invalid_cases(ctx):
    X = ctx["X"].copy()
    X["Amount"] = 0.0
    Xp, n_invalid = perturb(X, ["Amount"], "relative", 0.05)
    assert n_invalid == 0 and (Xp["Amount"] >= 0).all()
    X2 = ctx["X"].copy()
    X2.iloc[:10, X2.columns.get_loc("Amount")] = 1.0
    Xp2, _ = perturb(X2, ["Amount"], "gaussian", 50.0, scale=pd.Series({"Amount": 1.0}), seed=0)
    assert (Xp2["Amount"] >= 0).all()


def test_perturbation_does_not_modify_input(ctx):
    before = ctx["X"].copy()
    perturb(ctx["X"], V_COLS, "gaussian", 0.1, scale=ctx["std"])
    pd.testing.assert_frame_equal(ctx["X"], before)


def test_unknown_perturbation_is_rejected(ctx):
    with pytest.raises(ValueError, match="Perturbation inconnue"):
        perturb(ctx["X"], ["V1"], "uniforme", 0.1)


# ---------------------------------------- propriété : entrée valide => probabilité valide
@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(st.floats(-50, 50, allow_nan=False, allow_infinity=False), min_size=28, max_size=28),
    amount=st.floats(0, 1e7, allow_nan=False, allow_infinity=False),
    time=st.floats(0, 2e5, allow_nan=False, allow_infinity=False),
)
def test_t04_property_any_valid_transaction_gets_valid_probability(ctx, values, amount, time):
    row = pd.DataFrame([[time, *values, amount]], columns=FEATURES)
    y_pred, proba = predict_fraud(ctx["model"], row, threshold=ctx["threshold"])
    assert proba.shape == (1,) and 0.0 <= proba[0] <= 1.0
    assert y_pred[0] in (0, 1)

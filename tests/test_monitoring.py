"""Tests de la surveillance après déploiement (Partie X) : dérive des données (PSI) et taux de signalement.

Ces indicateurs ne nécessitent pas de labels : ils sont disponibles immédiatement en exploitation.
Ils corrigent l'anomalie A4 (décalage dans la plage d'entraînement non détecté par les alertes de plage).
"""

from pathlib import Path

import numpy as np
import pytest

from app.prediction import (
    PSI_ALERT,
    PSI_WARNING,
    check_alerts,
    drift_reference,
    monitoring_report,
    population_stability_index,
    predict_fraud,
    training_reference,
)
from app.preprocessing import FEATURES
from app.training import run_training_pipeline

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "creditcard_sample.csv"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    run = run_training_pipeline(SAMPLE, tmp_path_factory.mktemp("mon") / "model.joblib")
    s = run["splits"]
    y_pred_val, _ = predict_fraud(run["pipeline"], s["X_val"], threshold=run["threshold"])
    return {
        "run": run,
        "X_train": s["X_train"][FEATURES],
        "X_val": s["X_val"][FEATURES],
        "drift_ref": drift_reference(s["X_train"]),
        "expected_rate": float(y_pred_val.mean()),
    }


def test_reference_has_ten_bins_per_feature(ctx):
    ref = ctx["drift_ref"]
    assert set(ref) == set(FEATURES) - {"Time"}
    assert all(abs(sum(r["ref"]) - 1) < 1e-9 for r in ref.values())
    assert np.isinf(ref["V1"]["edges"][0]) and np.isinf(ref["V1"]["edges"][-1])


def test_identical_distribution_has_zero_psi(ctx):
    psi = population_stability_index(ctx["X_train"], ctx["drift_ref"])
    assert psi.max() < 1e-9


def test_same_population_is_stable(ctx):
    psi = population_stability_index(ctx["X_val"], ctx["drift_ref"])
    assert psi.max() < PSI_WARNING


def test_a4_in_range_shift_is_detected_by_psi(ctx):
    """V14 décalé d'un écart-type : invisible pour l'alerte de plage, détecté par le PSI."""
    shifted = ctx["X_val"].assign(V14=ctx["X_val"]["V14"] + ctx["X_train"]["V14"].std())
    range_alerts = [a for a in check_alerts(shifted, training_reference(ctx["X_train"])) if a.startswith("derive:")]
    psi = population_stability_index(shifted, ctx["drift_ref"])
    assert psi["V14"] > PSI_ALERT
    assert psi.idxmax() == "V14"
    assert not range_alerts or "V14" not in range_alerts[0]


def test_monitoring_report_without_drift(ctx):
    run = ctx["run"]
    y_pred, _ = predict_fraud(run["pipeline"], ctx["X_val"], threshold=run["threshold"])
    report = monitoring_report(ctx["X_val"], y_pred, ctx["drift_ref"], ctx["expected_rate"])
    assert report["alerts"] == []
    assert report["flag_rate_ratio"] == pytest.approx(1.0)


def test_monitoring_report_flags_abnormal_signal_rate(ctx):
    y_pred = np.ones(len(ctx["X_val"]), dtype=int)  # tout est signalé : panne ou attaque
    report = monitoring_report(ctx["X_val"], y_pred, ctx["drift_ref"], ctx["expected_rate"])
    assert any(a.startswith("taux_signalement") for a in report["alerts"])


def test_monitoring_report_flags_drift(ctx):
    shifted = ctx["X_val"].assign(V14=ctx["X_val"]["V14"] + ctx["X_train"]["V14"].std())
    y_pred, _ = predict_fraud(ctx["run"]["pipeline"], shifted, threshold=ctx["run"]["threshold"])
    report = monitoring_report(shifted, y_pred, ctx["drift_ref"], ctx["expected_rate"])
    assert "derive_importante:V14" in report["alerts"]

"""Tests de reproductibilité et de non-régression (Partie VI).

La référence `reports/baseline_ci.json` est créée UNE FOIS, manuellement, à partir de la version v1 et
de l'échantillon versionné ; elle n'est jamais recalculée par les tests (sinon les régressions seraient masquées).
Exigences couvertes : T06, T16 (docs/plan_de_test.md).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.evaluation import REGRESSION_TOLERANCES, compare_to_baseline
from app.training import CURRENT_VERSION, compute_reference, environment_info, file_sha256, main, run_training_pipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "creditcard_sample.csv"
BASELINE = ROOT / "reports" / "baseline_ci.json"


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.is_file():
        pytest.fail(f"Référence absente : {BASELINE}. La créer manuellement (voir README).")
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def current():
    return compute_reference(SAMPLE, CURRENT_VERSION)


# ---------------------------------------------------------------- T06 reproductibilité
def test_t06_two_runs_give_identical_results(tmp_path):
    first = run_training_pipeline(SAMPLE, tmp_path / "a.joblib")
    second = run_training_pipeline(SAMPLE, tmp_path / "b.joblib")
    assert first["threshold"] == second["threshold"]
    assert first["params"] == second["params"]
    assert np.max(np.abs(first["proba_test"] - second["proba_test"])) <= 1e-9


# ---------------------------------------------------------------- contenu de la référence
def test_baseline_records_everything_required(baseline):
    for key in (
        "version",
        "code_version",
        "params",
        "environment",
        "seed",
        "threshold",
        "dataset",
        "validation_metrics",
    ):
        assert key in baseline, f"champ manquant dans la référence : {key}"
    assert {"sha256", "rows", "fraud_rate"} <= baseline["dataset"].keys()
    assert {"pr_auc", "recall", "precision", "fpr"} <= baseline["validation_metrics"].keys()


def test_baseline_refers_to_the_versioned_sample(baseline):
    assert baseline["dataset"]["sha256"] == file_sha256(SAMPLE), "l'échantillon a changé depuis la référence"


def test_environment_matches_baseline(baseline):
    env = environment_info()
    for lib in ("scikit-learn", "numpy", "pandas"):
        assert env[lib] == baseline["environment"][lib], f"version de {lib} différente de la référence"


# ---------------------------------------------------------------- T16 non-régression
def test_t16_current_version_does_not_regress(baseline, current):
    comparison = compare_to_baseline(current["validation_metrics"], baseline["validation_metrics"])
    regressions = {m: d for m, d in comparison["metrics"].items() if d["regression"]}
    assert comparison["decision"] == "accepter", (
        f"Version {CURRENT_VERSION} vs référence {baseline['version']} : {comparison['decision']} — {regressions}"
    )


def test_baseline_is_never_recomputed_by_tests(baseline, current):
    assert json.loads(BASELINE.read_text()) == baseline


def test_baseline_creation_refuses_to_overwrite(tmp_path):
    existing = tmp_path / "ref.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit, match="jamais recalculée"):
        main(["--data", str(SAMPLE), "--output", str(existing)])
    assert existing.read_text() == "{}"


# ---------------------------------------------------------------- logique de décision
REF = {"pr_auc": 0.80, "recall": 0.85, "precision": 0.80, "fpr": 0.0005}


@pytest.mark.parametrize(
    "new, decision",
    [
        (REF, "accepter"),
        (REF | {"pr_auc": 0.85, "fpr": 0.0001}, "accepter"),  # amélioration
        (REF | {"pr_auc": 0.79}, "accepter"),  # dégradation dans la tolérance
        (REF | {"pr_auc": 0.77}, "investiguer"),  # au-delà de la tolérance
        (REF | {"recall": 0.75}, "rejeter"),  # au-delà du double de la tolérance
        (REF | {"fpr": 0.002}, "rejeter"),
    ],
)
def test_compare_to_baseline_decision(new, decision):
    result = compare_to_baseline(new, REF)
    assert result["decision"] == decision
    assert set(result["metrics"]) == set(REGRESSION_TOLERANCES)

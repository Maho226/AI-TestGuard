"""Tests de performance (Partie VIII) — critères du plan de test, § 5 (T17).

Exécutés sur le modèle entraîné avec l'échantillon : les seuils ont une marge confortable car les
machines d'intégration continue sont partagées. Les mesures officielles (protocole complet, modèle
du dataset complet) sont produites par le notebook 05_performance.ipynb.
"""

from pathlib import Path

import pytest

from app.prediction import measure_performance
from app.training import run_training_pipeline

# tests coûteux : exclus des Pull Requests, lancés manuellement ou chaque semaine (workflow)
pytestmark = pytest.mark.slow

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "creditcard_sample.csv"

P95_SINGLE_MS = 50.0
MIN_THROUGHPUT = 1_000
MAX_MODEL_MB = 10.0
MAX_EXTRA_MEMORY_MB = 200.0


@pytest.fixture(scope="module")
def perf(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("perf") / "model.joblib"
    run = run_training_pipeline(SAMPLE, model_path)
    measures = measure_performance(
        run["pipeline"],
        run["splits"]["X_val"],
        run["threshold"],
        n_single=300,
        warmup=20,
        batch_sizes=(100, 1000),
        n_batch_repeats=3,
    )
    return measures | {"model_mb": model_path.stat().st_size / 1e6}


def test_t17_single_prediction_p95_latency(perf):
    assert perf["latence_prediction_isolee_avec_controle_entrees"]["p95_ms"] <= P95_SINGLE_MS


def test_t17_input_validation_overhead_is_measured_separately(perf):
    model = perf["latence_prediction_isolee_modele_seul"]["p50_ms"]
    full = perf["latence_prediction_isolee_avec_controle_entrees"]["p50_ms"]
    assert model > 0 and full >= model


def test_t17_batch_throughput(perf):
    assert perf["debit_par_lot"]["1000"]["predictions_par_s"] >= MIN_THROUGHPUT


def test_t17_model_size(perf):
    assert perf["model_mb"] <= MAX_MODEL_MB


def test_t17_inference_memory(perf):
    assert perf["memoire_pic_inference_lot_mo"] <= MAX_EXTRA_MEMORY_MB


def test_protocol_is_documented(perf):
    proto = perf["protocole"]
    assert proto["repetitions_isolees"] == 300 and proto["horloge"] == "time.perf_counter"
    assert {"processeur", "coeurs_logiques", "memoire_totale_go"} <= proto["materiel"].keys()

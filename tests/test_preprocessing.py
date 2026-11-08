"""Tests unitaires du prétraitement et de la qualité des données (Partie II).

Exigences couvertes : T01, T02, T03, T08, T09, T11 (docs/plan_de_test.md).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.preprocessing import StandardScaler

from app.preprocessing import (
    FEATURES,
    TARGET,
    load_data,
    preprocess_data,
    remove_duplicates,
    split_data,
    validate_data,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "creditcard_sample.csv"


# ---------------------------------------------------------------- T01 colonnes
def test_t01_valid_data_is_accepted(valid_df):
    validated = validate_data(valid_df)
    assert list(validated.columns) == FEATURES + [TARGET]
    assert len(validated) == len(valid_df)


@pytest.mark.parametrize("column", ["Time", "V1", "V14", "V28", "Amount", "Class"])
def test_t01_missing_column_is_rejected(valid_df, column):
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        validate_data(valid_df.drop(columns=column))


def test_t01_target_not_required_for_prediction(valid_df):
    validated = validate_data(valid_df.drop(columns=TARGET), require_target=False)
    assert TARGET not in validated.columns


def test_t09_duplicated_column_is_rejected(valid_df):
    duplicated = pd.concat([valid_df, valid_df[["V1"]]], axis=1)
    with pytest.raises(ValueError, match="Colonnes dupliquées"):
        validate_data(duplicated)


def test_t09_unknown_column_is_rejected(valid_df):
    with pytest.raises(ValueError, match="Colonnes inconnues"):
        validate_data(valid_df.assign(card_number="4970000000000000"))


# ---------------------------------------------------------------- T02 types
def test_t02_non_numeric_column_is_rejected(valid_df):
    df = valid_df.astype({"V1": "object"})
    df.loc[0, "V1"] = "abc"
    with pytest.raises(ValueError, match="non numériques"):
        validate_data(df)


def test_t02_boolean_column_is_rejected(valid_df):
    with pytest.raises(ValueError, match="non numériques"):
        validate_data(valid_df.assign(Amount=True))


# ---------------------------------------------------------------- T09 contenu invalide
def test_t09_empty_dataframe_is_rejected():
    with pytest.raises(ValueError, match="Données vides"):
        validate_data(pd.DataFrame(columns=FEATURES + [TARGET]))


def test_t09_non_dataframe_is_rejected(valid_df):
    with pytest.raises(ValueError, match="DataFrame"):
        validate_data(valid_df.to_numpy())


@pytest.mark.parametrize("column", ["Time", "V7", "Amount"])
def test_t09_missing_values_are_rejected(valid_df, column):
    valid_df.loc[3, column] = np.nan
    with pytest.raises(ValueError, match=f"Valeurs manquantes dans : \\['{column}'\\]"):
        validate_data(valid_df)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_t09_infinite_values_are_rejected(valid_df, value):
    valid_df.loc[5, "V3"] = value
    with pytest.raises(ValueError, match="Valeurs infinies dans : \\['V3'\\]"):
        validate_data(valid_df)


def test_t09_negative_amount_is_rejected(valid_df):
    valid_df.loc[0, "Amount"] = -10.0
    with pytest.raises(ValueError, match="négatif"):
        validate_data(valid_df)


def test_t09_zero_amount_is_accepted(valid_df):
    """Valeur limite : un montant nul est valide."""
    valid_df.loc[0, "Amount"] = 0.0
    assert validate_data(valid_df).loc[0, "Amount"] == 0.0


# ---------------------------------------------------------------- T08 cible
@pytest.mark.parametrize("bad_value", [2, -1, 0.5])
def test_t08_invalid_target_is_rejected(valid_df, bad_value):
    valid_df[TARGET] = valid_df[TARGET].astype(float)
    valid_df.loc[0, TARGET] = bad_value
    with pytest.raises(ValueError, match="cible invalides"):
        validate_data(valid_df)


# ---------------------------------------------------------------- T03 sortie du prétraitement
def test_t03_preprocess_output_contract(valid_df):
    X, y = preprocess_data(valid_df)
    assert X.shape == (len(valid_df), 30)
    assert list(X.columns) == FEATURES
    assert (X.dtypes == "float64").all()
    assert np.isfinite(X.to_numpy()).all()
    assert y.dtype == "int64" and set(y.unique()) <= {0, 1}


def test_t03_column_order_is_normalized(valid_df):
    shuffled = valid_df[list(reversed(valid_df.columns))]
    X, _ = preprocess_data(shuffled)
    assert list(X.columns) == FEATURES


def test_t03_single_observation_is_processed(valid_df):
    X, y = preprocess_data(valid_df.iloc[[0]])
    assert X.shape == (1, 30)
    assert len(y) == 1


def test_t03_input_is_not_modified(valid_df):
    before = valid_df.copy()
    preprocess_data(valid_df)
    pd.testing.assert_frame_equal(valid_df, before)


@settings(max_examples=50, deadline=None)
@given(arrays(np.float64, (5, 30), elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)))
def test_t03_property_finite_inputs_give_finite_outputs(values):
    df = pd.DataFrame(values, columns=FEATURES).assign(Amount=lambda d: d["Amount"].abs())
    X, _ = preprocess_data(df, require_target=False)
    assert X.shape == (5, 30)
    assert np.isfinite(X.to_numpy()).all()


# ---------------------------------------------------------------- doublons
def test_remove_duplicates_counts_removed_rows(valid_df):
    with_dups = pd.concat([valid_df, valid_df.iloc[:7]], ignore_index=True)
    deduplicated, removed = remove_duplicates(with_dups)
    assert removed == 7
    assert len(deduplicated) == len(valid_df)
    assert not deduplicated.duplicated().any()


# ---------------------------------------------------------------- T11 séparation
def test_t11_split_sizes_stratification_and_disjointness(valid_df):
    X, y = preprocess_data(valid_df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y)
    n = len(X)
    assert abs(len(X_tr) - 0.6 * n) <= 1 and abs(len(X_va) - 0.2 * n) <= 1 and abs(len(X_te) - 0.2 * n) <= 1
    for part in (y_tr, y_va, y_te):
        assert abs(part.mean() - y.mean()) <= 0.02
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert set(X_tr.index).isdisjoint(X_va.index)
    assert set(X_va.index).isdisjoint(X_te.index)


def test_t11_split_is_reproducible(valid_df):
    X, y = preprocess_data(valid_df)
    first = split_data(X, y)
    second = split_data(X, y)
    for a, b in zip(first, second, strict=True):
        assert list(a.index) == list(b.index)


# ---------------------------------------------------------------- chargement
def test_load_data_missing_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_data(tmp_path / "absent.csv")


def test_load_data_empty_file_is_rejected(tmp_path):
    empty = tmp_path / "vide.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="illisible"):
        load_data(empty)


def test_t01_versioned_sample_respects_contract():
    df = load_data(SAMPLE)
    assert df.shape == (5492, 31)
    assert int(df[TARGET].sum()) == 492


# ---------------------------------------------------------------- exemple de l'énoncé
def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)

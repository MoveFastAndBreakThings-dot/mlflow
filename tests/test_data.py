import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.preprocess import load_raw, preprocess

DATA_PATH = "data/raw/drug_discovery_dataset.csv"
TARGET = "clinical_trial_phase_prediction"


@pytest.fixture(scope="module")
def raw_df():
    return load_raw(DATA_PATH)


@pytest.fixture(scope="module")
def splits(raw_df):
    return preprocess(raw_df.copy(), TARGET)


# --- raw data ---

def test_row_count(raw_df):
    assert raw_df.shape[0] == 2800

def test_expected_columns(raw_df):
    expected = [
        "molecule_id", "molecular_weight", "logP",
        "num_hydrogen_bond_donors", "num_hydrogen_bond_acceptors",
        "topological_polar_surface_area", "rotatable_bonds",
        "aromatic_rings", "bioactivity_score", "toxicity_risk_level",
        "drug_likeness", "binding_affinity_kcal_mol",
        "clinical_trial_phase_prediction",
    ]
    assert list(raw_df.columns) == expected

def test_no_missing_values(raw_df):
    assert raw_df.isnull().sum().sum() == 0

def test_target_classes(raw_df):
    expected_classes = {"Preclinical", "Phase I", "Phase II", "Phase III"}
    assert set(raw_df[TARGET].unique()) == expected_classes

def test_target_distribution(raw_df):
    counts = raw_df[TARGET].value_counts()
    assert counts["Preclinical"] > counts["Phase III"]   # expected imbalance


# --- preprocessing ---

def test_split_sizes(splits):
    X_train, X_test, y_train, y_test, _, _ = splits
    assert len(X_train) + len(X_test) == 2800

def test_split_ratio(splits):
    X_train, X_test, _, _, _, _ = splits
    ratio = len(X_test) / (len(X_train) + len(X_test))
    assert abs(ratio - 0.2) < 0.01

def test_feature_columns_match(splits):
    X_train, X_test, _, _, _, _ = splits
    assert list(X_train.columns) == list(X_test.columns)

def test_no_target_in_features(splits):
    X_train, _, _, _, _, _ = splits
    assert TARGET not in X_train.columns

def test_no_missing_after_preprocess(splits):
    X_train, X_test, y_train, y_test, _, _ = splits
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0

def test_test_labels_subset_of_train(splits):
    _, _, y_train, y_test, _, _ = splits
    assert set(y_test.unique()).issubset(set(y_train.unique()))

def test_features_are_scaled(splits):
    X_train, _, _, _, _, _ = splits
    # StandardScaler: mean ≈ 0, std ≈ 1
    assert abs(X_train["molecular_weight"].mean()) < 0.1
    assert abs(X_train["molecular_weight"].std() - 1.0) < 0.1

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.preprocess import load_raw, preprocess
from src.models.train import get_models
import yaml

DATA_PATH = "data/raw/drug_discovery_dataset.csv"
TARGET = "clinical_trial_phase_prediction"


@pytest.fixture(scope="module")
def cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def splits():
    df = load_raw(DATA_PATH)
    return preprocess(df, TARGET)


@pytest.fixture(scope="module")
def trained_models(cfg, splits):
    X_train, X_test, y_train, y_test, _, _ = splits
    models = get_models(cfg)
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = (model, X_test, y_test)
    return trained


@pytest.mark.parametrize("model_name", [
    "logistic_regression", "random_forest", "xgboost",
    "lightgbm", "svm", "knn", "gradient_boosting"
])
def test_model_predicts(trained_models, model_name):
    model, X_test, _ = trained_models[model_name]
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


@pytest.mark.parametrize("model_name", [
    "logistic_regression", "random_forest", "xgboost",
    "lightgbm", "svm", "knn", "gradient_boosting"
])
def test_predictions_valid_classes(trained_models, model_name):
    model, X_test, y_test = trained_models[model_name]
    preds = model.predict(X_test)
    valid_classes = set(y_test.unique())
    assert set(preds).issubset(valid_classes)


@pytest.mark.parametrize("model_name", [
    "logistic_regression", "random_forest", "xgboost",
    "lightgbm", "svm", "knn", "gradient_boosting"
])
def test_model_beats_random(trained_models, model_name):
    from sklearn.metrics import f1_score
    model, X_test, y_test = trained_models[model_name]
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average="weighted")
    random_baseline = 0.20   # conservative floor — balanced class_weight trades accuracy for recall
    assert f1 > random_baseline, f"{model_name} f1_weighted {f1:.3f} ≤ floor {random_baseline}"


@pytest.mark.parametrize("model_name", [
    "logistic_regression", "random_forest", "xgboost",
    "lightgbm", "svm", "knn", "gradient_boosting"
])
def test_predict_proba_shape(trained_models, model_name):
    model, X_test, y_test = trained_models[model_name]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 4)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

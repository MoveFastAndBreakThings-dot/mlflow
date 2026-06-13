import pytest
import mlflow
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "drug-trial-phase-classifier"
EXPECTED_MODELS = {
    "logistic_regression", "random_forest", "xgboost",
    "lightgbm", "svm", "knn", "gradient_boosting"
}
EXPECTED_METRICS = {"accuracy", "f1_weighted", "precision_weighted", "recall_weighted"}


@pytest.fixture(scope="module")
def client():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.MlflowClient()


@pytest.fixture(scope="module")
def all_runs(client):
    experiments = client.search_experiments()
    exp = next((e for e in experiments if e.name == "drug-discovery-tournament"), None)
    assert exp is not None, "Experiment 'drug-discovery-tournament' not found — run pipeline.py first"
    return client.search_runs(experiment_ids=[exp.experiment_id])


def test_experiment_exists(client):
    experiments = client.search_experiments()
    names = [e.name for e in experiments]
    assert "drug-discovery-tournament" in names

def test_all_models_ran(all_runs):
    model_types = {r.data.tags.get("model_type") for r in all_runs}
    assert EXPECTED_MODELS.issubset(model_types), \
        f"Missing runs for: {EXPECTED_MODELS - model_types}"

def test_all_runs_finished(all_runs):
    statuses = {r.info.status for r in all_runs}
    assert statuses == {"FINISHED"}, f"Some runs not finished: {statuses}"

def test_metrics_logged(all_runs):
    for run in all_runs:
        missing = EXPECTED_METRICS - set(run.data.metrics.keys())
        assert not missing, f"Run {run.data.tags.get('model_type')} missing metrics: {missing}"

def test_metrics_in_valid_range(all_runs):
    for run in all_runs:
        for metric in EXPECTED_METRICS:
            val = run.data.metrics[metric]
            assert 0.0 <= val <= 1.0, \
                f"{run.data.tags.get('model_type')}.{metric} = {val} out of range"

def test_model_registered(client):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    assert len(versions) >= 1, "No model versions registered"

def test_exactly_one_production_model(client):
    versions = [
        v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
        if v.current_stage == "Production"
    ]
    assert len(versions) == 1, f"Expected 1 Production model, got {len(versions)}"

def test_production_model_has_best_f1(client, all_runs):
    prod = next(
        v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
        if v.current_stage == "Production"
    )
    prod_run = next(r for r in all_runs if r.info.run_id == prod.run_id)
    prod_f1 = prod_run.data.metrics["f1_weighted"]
    for run in all_runs:
        assert run.data.metrics["f1_weighted"] <= prod_f1 + 1e-6, \
            f"{run.data.tags.get('model_type')} f1={run.data.metrics['f1_weighted']:.4f} beats Production f1={prod_f1:.4f}"

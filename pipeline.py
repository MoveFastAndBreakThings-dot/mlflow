"""
Main pipeline: train all models, track with MLflow, promote winner to Production.
Run: python pipeline.py
"""
import os
import tempfile
import yaml
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.data.preprocess import load_raw, preprocess
from src.models.train import get_models
from src.evaluation.evaluate import compute_metrics, plot_confusion_matrix, print_report


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_tournament(cfg: dict):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(cfg["experiment"]["name"])
    client = MlflowClient()

    target = cfg["experiment"]["target"]
    df = load_raw(cfg["data"]["raw_path"])
    X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess(
        df, target, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )

    class_labels = label_encoder.classes_
    models = get_models(cfg)
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("target", target)

            # log hyperparams
            if model_name in cfg["models"]:
                mlflow.log_params(cfg["models"][model_name])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)
            mlflow.log_metrics(metrics)
            print_report(y_test, y_pred, model_name)

            # log confusion matrix as artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                cm_path = os.path.join(tmpdir, f"confusion_matrix_{model_name}.png")
                plot_confusion_matrix(y_test, y_pred, class_labels, model_name, cm_path)
                mlflow.log_artifact(cm_path, artifact_path="plots")

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
embatk_love-em@DESKTOP-8HB69FL:/mnt/e/terminal based website$ cd /mnt/e/terminal\ based\ website/ssh-portfolio && fly deploy
Welcome back!
Your session has expired, please log in to continue using flyctl.

? Would you like to sign in? Yes
failed opening browser. Copy the url (https://fly.io/app/auth/cli/37626c68767936336e667a7866676b62636e73626a6f6e72616b3363646b6172) into a browser and continue
Opening https://fly.io/app/auth/cli/37626c68767936336e667a7866676b62636e73626a6f6e72616b3363646b6172 ...

Waiting for session...⣾Error: Login expired, please try again                registered_model_name=cfg["registry"]["model_name"],
            )

            results[model_name] = {
                "run_id": run.info.run_id,
                "metrics": metrics,
            }

    promote_winner(client, results, cfg)


def promote_winner(client: MlflowClient, results: dict, cfg: dict):
    metric = cfg["registry"]["promotion_metric"]
    winner_name = max(results, key=lambda m: results[m]["metrics"][metric])
    winner = results[winner_name]

    print(f"\nWinner: {winner_name} ({metric}={winner['metrics'][metric]:.4f})")

    # find model version registered from winner run
    model_name = cfg["registry"]["model_name"]
    versions = client.search_model_versions(f"name='{model_name}'")
    winner_version = next(
        (v for v in versions if v.run_id == winner["run_id"]), None
    )

    if not winner_version:
        print("Could not find registered version for winner run.")
        return

    # archive current Production if exists
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    for v in prod_versions:
        client.transition_model_version_stage(
            name=model_name, version=v.version, stage="Archived"
        )
        print(f"Archived previous Production v{v.version}")

    # promote winner to Production
    client.transition_model_version_stage(
        name=model_name, version=winner_version.version, stage="Production"
    )
    print(f"Promoted {winner_name} v{winner_version.version} to Production")


if __name__ == "__main__":
    cfg = load_config()
    run_tournament(cfg)

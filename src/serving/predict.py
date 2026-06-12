import mlflow.pyfunc
import pandas as pd


def load_production_model(model_name: str):
    model_uri = f"models:/{model_name}/Production"
    return mlflow.pyfunc.load_model(model_uri)


def predict(model, features: dict) -> list:
    df = pd.DataFrame([features])
    return model.predict(df).tolist()

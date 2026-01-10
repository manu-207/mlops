import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import mlflow
import os

# ================= Load Params =================
params = yaml.safe_load(open("params.yaml"))["train"]

# ================= MLflow Config (ENV BASED) =================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "manu7-mlops")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def evaluate(data_path, target, model_path):

    data = pd.read_csv(data_path)

    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(data.columns)}"
        )

    X = data.drop(columns=[target])
    y = data[target]

    # Load model (tracked by DVC locally)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run(run_name="evaluation"):

        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)

        mlflow.log_metric("evaluation_accuracy", accuracy)

        print(f"Model accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate(
        params["data"],
        params["target"],
        params["model"]
    )

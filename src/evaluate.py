import os
import pickle
import yaml
import pandas as pd
import mlflow

from sklearn.metrics import accuracy_score

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# ================= Load Params =================
params = yaml.safe_load(open("params.yaml"))["train"]

REFERENCE_DATA_PATH = "data/processed/reference.csv"
CURRENT_DATA_PATH = "data/processed/current.csv"
REPORT_PATH = "reports/evidently_data_drift.html"

TARGET = params["target"]
MODEL_PATH = params["model"]


# ================= MLflow Config =================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "manu7-mlops")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def evaluate():

    # ---------- Load Data ----------
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    current_data = pd.read_csv(CURRENT_DATA_PATH)

    # ---------- Validate Target ----------
    if TARGET not in reference_data.columns:
        raise ValueError(f"Target column '{TARGET}' not found in reference data")

    if TARGET not in current_data.columns:
        raise ValueError(f"Target column '{TARGET}' not found in current data")

    # ---------- Split Features & Target ----------
    X_reference = reference_data.drop(columns=[TARGET])
    X_current = current_data.drop(columns=[TARGET])

    y_current = current_data[TARGET]

    # ---------- Validate Feature Consistency ----------
    if list(X_reference.columns) != list(X_current.columns):
        raise ValueError("Feature columns mismatch between reference and current data")

    # ---------- Load Model ----------
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # ---------- MLflow Evaluation ----------
    with mlflow.start_run(run_name="evaluation"):

        predictions = model.predict(X_current)
        accuracy = accuracy_score(y_current, predictions)

        mlflow.log_metric("evaluation_accuracy", accuracy)
        print(f"Model accuracy: {accuracy}")

        # ---------- Evidently Data Drift (FEATURES ONLY) ----------
        report = Report(
            metrics=[
                DataDriftPreset()
            ]
        )

        report.run(
            reference_data=X_reference,
            current_data=X_current
        )

        os.makedirs("reports", exist_ok=True)
        report.save_html(REPORT_PATH)

        # Log Evidently report to MLflow
        mlflow.log_artifact(REPORT_PATH)

        print(f"Evidently report saved at: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()

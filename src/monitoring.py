import os
import pickle
import pandas as pd
import mlflow

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

REFERENCE_DATA_PATH = "data/processed/reference.csv"
CURRENT_DATA_PATH = "data/processed/current.csv"
MODEL_PATH = "models/model.pkl"
TARGET = "Outcome"

# ================= MLflow Config =================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "manu7-mlops"))


def monitor():

    reference = pd.read_csv(REFERENCE_DATA_PATH)
    current = pd.read_csv(CURRENT_DATA_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    reference["prediction"] = model.predict(reference.drop(columns=[TARGET]))
    current["prediction"] = model.predict(current.drop(columns=[TARGET]))

    with mlflow.start_run(run_name="monitoring"):

        report = Report(
            metrics=[
                DataDriftPreset(),
                ClassificationPreset()
            ]
        )

        report.run(reference_data=reference, current_data=current)

        os.makedirs("reports", exist_ok=True)
        report_path = "reports/evidently_report.html"
        report.save_html(report_path)

        mlflow.log_artifact(report_path, artifact_path="evidently")

        print("✅ Evidently monitoring report logged to MLflow")


if __name__ == "__main__":
    monitor()

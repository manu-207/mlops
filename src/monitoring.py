import pandas as pd
import pickle
import yaml
import mlflow
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# ================= Load Params =================
params = yaml.safe_load(open("params.yaml"))["monitor"]
train_params = yaml.safe_load(open("params.yaml"))["train"]

# ================= MLflow Config =================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "manu7-mlops")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def monitor(reference_path, current_path, model_path, target):

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Split features & target
    X_ref = reference.drop(columns=[target])
    y_ref = reference[target]

    X_cur = current.drop(columns=[target])
    y_cur = current[target]

    # Predictions
    reference["prediction"] = model.predict(X_ref)
    current["prediction"] = model.predict(X_cur)

    with mlflow.start_run(run_name="monitoring"):

        # ================= Evidently Report =================
        report = Report(
            metrics=[
                DataDriftPreset(),
                ClassificationPreset()
            ]
        )

        report.run(
            reference_data=reference,
            current_data=current
        )

        os.makedirs("reports", exist_ok=True)
        report_path = "reports/evidently_report.html"
        report.save_html(report_path)

        # ================= Log to MLflow =================
        mlflow.log_artifact(report_path, artifact_path="evidently")

        print("✅ Evidently monitoring report generated and logged to MLflow")


# ================= Run =================
if __name__ == "__main__":
    monitor(
        params["reference_data"],
        params["current_data"],
        train_params["model"],
        train_params["target"]
    )

import os
import yaml
import pickle
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
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "manu7-mlops"))


def evaluate():

    reference_data = pd.read_csv(REFERENCE_DATA_PATH)

    # 🔥 Generate current data dynamically (runtime only)
    current_data = reference_data.sample(frac=0.2, random_state=42)
    os.makedirs("data/processed", exist_ok=True)
    current_data.to_csv(CURRENT_DATA_PATH, index=False)

    X_ref = reference_data.drop(columns=[TARGET])
    X_cur = current_data.drop(columns=[TARGET])
    y_cur = current_data[TARGET]

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run(run_name="evaluation"):

        preds = model.predict(X_cur)
        acc = accuracy_score(y_cur, preds)
        mlflow.log_metric("evaluation_accuracy", acc)

        print(f"✅ Evaluation Accuracy: {acc}")

        # ================= Evidently Drift =================
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=X_ref, current_data=X_cur)

        os.makedirs("reports", exist_ok=True)
        report.save_html(REPORT_PATH)

        mlflow.log_artifact(REPORT_PATH)

        print("✅ Evidently drift report logged to MLflow")


if __name__ == "__main__":
    evaluate()

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
    current_data = pd.read_csv(CURRENT_DATA_PATH)

    # ---------- Validate ----------
    if TARGET not in reference_data.columns or TARGET not in current_data.columns:
        raise ValueError("Target column missing in datasets")

    X_ref = reference_data.drop(columns=[TARGET])
    X_cur = current_data.drop(columns=[TARGET])
    y_cur = current_data[TARGET]

    if list(X_ref.columns) != list(X_cur.columns):
        raise ValueError("Feature mismatch between reference & current data")

    # ---------- Load Model ----------
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run(run_name="evaluation"):

        preds = model.predict(X_cur)
        acc = accuracy_score(y_cur, preds)

        mlflow.log_metric("evaluation_accuracy", acc)
        print(f"✅ Evaluation Accuracy: {acc}")

        # ---------- Evidently Drift Report ----------
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=X_ref,
            current_data=X_cur
        )

        os.makedirs("reports", exist_ok=True)
        report.save_html(REPORT_PATH)

        # 🔥 This makes it open inside MLflow UI
        mlflow.log_artifact(REPORT_PATH)

        print("✅ Evidently report logged to MLflow")

if __name__ == "__main__":
    evaluate()

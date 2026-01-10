import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import mlflow

# ================= Load Params =================
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, target, model_path):

    data = pd.read_csv(data_path)

    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(data.columns)}"
        )

    X = data.drop(columns=[target])
    y = data[target]

    # ===== MLflow EC2 Config =====
    mlflow.set_tracking_uri("http://13.233.85.40:5000")
    mlflow.set_experiment("manu7-mlops")

    # Load model (tracked by DVC locally)
    model = pickle.load(open(model_path, "rb"))

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

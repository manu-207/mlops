import os
import yaml
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature

# ================= Load Params =================
params = yaml.safe_load(open("params.yaml"))["train"]

# ================= MLflow Config =================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "manu7-mlops"))

# ================= Hyperparameter Tuning =================
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid

# ================= Training =================
def train():
    data = pd.read_csv(params["data"])
    target = params["target"]

    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in dataset")

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=params["random_state"]
    )

    with mlflow.start_run(run_name="training"):

        grid = hyperparameter_tuning(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # ---------- MLflow Logs ----------
        mlflow.log_metric("accuracy", accuracy)

        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)

        mlflow.log_text(
            str(confusion_matrix(y_test, y_pred)),
            artifact_file="confusion_matrix.txt"
        )

        mlflow.log_text(
            classification_report(y_test, y_pred),
            artifact_file="classification_report.txt"
        )

        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="manu7-rf-model"
        )

        # ---------- Save model locally (DVC) ----------
        os.makedirs(os.path.dirname(params["model"]), exist_ok=True)
        with open(params["model"], "wb") as f:
            pickle.dump(best_model, f)

        print(f"✅ Training completed | Accuracy: {accuracy}")

if __name__ == "__main__":
    train()

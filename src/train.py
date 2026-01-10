import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import mlflow.sklearn

# ================= Load Params =================
params = yaml.safe_load(open("params.yaml"))["train"]

# ================= MLflow Config  =================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "manu7-mlops")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# ================= Hyperparameter Tuning =================
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search

# ================= Training =================
def train(data_path, target, model_path, random_state, n_estimators, max_depth):

    data = pd.read_csv(data_path)

    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(data.columns)}"
        )

    X = data.drop(columns=[target])
    y = data[target]

    with mlflow.start_run(run_name="training"):

        # Log base params
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        signature = infer_signature(X_train, y_train)

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)

        for k, v in grid_search.best_params_.items():
            mlflow.log_param(f"best_{k}", v)

        mlflow.log_text(
            str(confusion_matrix(y_test, y_pred)),
            "confusion_matrix.txt"
        )

        mlflow.log_text(
            classification_report(y_test, y_pred),
            "classification_report.txt"
        )

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="manu7-rf-model"
        )

        # Save model for DVC
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Model saved at {model_path}")

# ================= Run =================
if __name__ == "__main__":
    train(
        params["data"],
        params["target"],
        params["model"],
        params["random_state"],
        params["n_estimators"],
        params["max_depth"]
    )

from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import mlflow.sklearn
import os

app = Flask(__name__)

# ================= MLflow Config =================
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://13.203.222.37:5000"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "manu7-rf-model"
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "prod"   # alias, NOT stage
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Load model using ALIAS
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.sklearn.load_model(model_uri)

# ================= Evidently Data Path =================
EVIDENTLY_DATA_PATH = "/opt/evidently/data"
CURRENT_DATA_FILE = f"{EVIDENTLY_DATA_PATH}/current.csv"

os.makedirs(EVIDENTLY_DATA_PATH, exist_ok=True)

# ================= Routes =================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        predictions = model.predict(df)

        # 🔥 LOG PRODUCTION DATA FOR EVIDENTLY
        df["prediction"] = predictions
        df.to_csv(
            CURRENT_DATA_FILE,
            mode="a",
            header=not os.path.exists(CURRENT_DATA_FILE),
            index=False
        )

        return jsonify({
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

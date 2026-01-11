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

MODEL_STAGE = os.getenv(
    "MODEL_STAGE",
    "Production"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model from MLflow Registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.sklearn.load_model(model_uri)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        predictions = model.predict(df)

        return jsonify({
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

import pandas as pd
import yaml
import os

# ================= Load Params =================
params = yaml.safe_load(open("params.yaml"))["preprocess"]

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

def preprocess(input_path: str, output_path: str):
    # Raw CSV has no headers
    data = pd.read_csv(input_path, header=None)
    data.columns = COLUMN_NAMES

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"✅ Preprocessed data saved at: {output_path}")

if __name__ == "__main__":
    preprocess(
        params["input"],
        params["output"]
    )

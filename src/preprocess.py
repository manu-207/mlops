import pandas as pd
import yaml
import os

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

REFERENCE_PATH = "data/processed/reference.csv"
CURRENT_PATH = "data/processed/current.csv"

def preprocess(input_path):
    # Load raw data
    data = pd.read_csv(input_path, header=None)
    data.columns = COLUMN_NAMES

    os.makedirs("data/processed", exist_ok=True)

    # 🔹 Split data for drift monitoring
    reference_data = data.sample(frac=0.7, random_state=42)
    current_data = data.drop(reference_data.index)

    reference_data.to_csv(REFERENCE_PATH, index=False)
    current_data.to_csv(CURRENT_PATH, index=False)

    print(f"✅ Reference data saved at: {REFERENCE_PATH}")
    print(f"✅ Current data saved at: {CURRENT_PATH}")

if __name__ == "__main__":
    preprocess(params["input"])

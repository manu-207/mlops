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

def preprocess(input_path, output_path):
    # RAW FILE HAS NO HEADERS → DEFINE THEM
    data = pd.read_csv(input_path, header=None)
    data.columns = COLUMN_NAMES

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess(params["input"], params["output"])

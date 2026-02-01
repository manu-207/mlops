import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import os

params = yaml.safe_load(open("params.yaml"))["split"]

data = pd.read_csv(params["input"])

reference, current = train_test_split(
    data,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

os.makedirs(os.path.dirname(params["reference"]), exist_ok=True)

reference.to_csv(params["reference"], index=False)
current.to_csv(params["current"], index=False)

print("✅ Reference and current data created")

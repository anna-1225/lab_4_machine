
import argparse
import os

import mlflow
from mlflow import MlflowClient

parser = argparse.ArgumentParser()

parser.add_argument("--version", required=True)
parser.add_argument("--alias", default="champion")

args = parser.parse_args()

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)

client = MlflowClient()

model_version = client.get_model_version(
    "heart-disease-classifier",
    args.version
)

client.set_registered_model_alias(
    "heart-disease-classifier",
    args.alias,
    args.version
)

print(
    f"{args.alias} -> version {model_version.version}, "
    f"run {model_version.run_id}"
)

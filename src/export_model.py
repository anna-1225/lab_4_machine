
import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/raw/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)

client = MlflowClient()

version = client.get_model_version_by_alias(
    "heart-disease-classifier",
    "champion"
)

model = mlflow.sklearn.load_model(
    f"models:/heart-disease-classifier/{version.version}"
)

export_path = ROOT / "exported_model"

if export_path.exists():
    shutil.rmtree(export_path)

mlflow.sklearn.save_model(
    model,
    str(export_path),
    pip_requirements=(
        ROOT / "requirements.txt"
    ).read_text().splitlines()
)

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=COLUMNS,
    na_values=["?"]
)

df = df.apply(pd.to_numeric, errors="raise")

X = df.drop(columns=["num"])
y = (df["num"] > 0).astype(int)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

metadata = {
    "name": "heart-disease-classifier",
    "version": str(version.version),
    "run_id": version.run_id,
    "features": X.columns.tolist(),
    "classes": ["absence", "presence"]
}

(export_path / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

request = {
    "instances": X_test.iloc[:3]
        .astype(object)
        .where(pd.notna(X_test.iloc[:3]), None)
        .to_dict(orient="records")
}

(ROOT / "sample_request.json").write_text(
    json.dumps(request, indent=2),
    encoding="utf-8"
)

predictions = model.predict(X_test.iloc[:3])

(ROOT / "expected_predictions.json").write_text(
    json.dumps(predictions.astype(int).tolist(), indent=2),
    encoding="utf-8"
)

print(json.dumps(metadata, indent=2))

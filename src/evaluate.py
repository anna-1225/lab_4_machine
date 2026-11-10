
import json
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/raw/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

model = mlflow.sklearn.load_model(
    str(ROOT / "exported_model")
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

predictions = model.predict(X_test)

metrics = {
    "test_accuracy": float(
        accuracy_score(y_test, predictions)
    ),
    "test_f1_macro": float(
        f1_score(y_test, predictions, average="macro")
    )
}

(ROOT / "test_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)

print(json.dumps(metrics, indent=2))

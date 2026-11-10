
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/raw/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--algorithm",
    choices=["lr", "rf"],
    required=True
)

parser.add_argument("--c", type=float, default=1.0)
parser.add_argument("--trees", type=int, default=100)
parser.add_argument("--depth", type=int, default=5)
parser.add_argument("--register", action="store_true")

args = parser.parse_args()

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)

mlflow.set_experiment("lab4-heart-disease")

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=COLUMNS,
    sep=",",
    na_values=["?"]
)

df = df.apply(pd.to_numeric, errors="raise")

if df["num"].isna().any():
    raise ValueError("Пропуски в целевой переменной")

X = df.drop(columns=["num"])
y = (df["num"] > 0).astype(int)

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
])

if args.algorithm == "lr":
    classifier = LogisticRegression(
        C=args.c,
        max_iter=2000,
        random_state=42
    )
else:
    classifier = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.depth,
        random_state=42,
        n_jobs=1
    )

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

data_hash = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

with mlflow.start_run(
    run_name=f"{args.algorithm}-experiment"
) as run:

    mlflow.log_params({
        "algorithm": args.algorithm,
        "random_state": 42,
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "n_test": len(X_test),
        "dataset": "UCI Processed Cleveland"
    })

    if args.algorithm == "lr":
        mlflow.log_param("C", args.c)
    else:
        mlflow.log_params({
            "trees": args.trees,
            "depth": args.depth
        })

    mlflow.set_tags({
        "purpose": "lab4-variant19",
        "dataset_sha256": data_hash,
        "selection_metric": "val_f1_macro"
    })

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    f1_macro = f1_score(
        y_val, predictions, average="macro"
    )

    mlflow.log_metrics({
        "val_accuracy": accuracy,
        "val_f1_macro": f1_macro
    })

    mlflow.log_input(
        mlflow.data.from_pandas(
            df,
            source=(
                "https://archive.ics.uci.edu/"
                "dataset/45/heart+disease"
            ),
            targets="num"
        ),
        context="full_dataset_before_split"
    )

    split_info = {
        "train_indices": X_train.index.tolist(),
        "validation_indices": X_val.index.tolist(),
        "test_indices": X_test.index.tolist(),
        "features": X.columns.tolist(),
        "target_definition": "num > 0",
        "random_state": 42
    }

    mlflow.log_dict(split_info, "split.json")

    mlflow.log_artifact(
        str(Path(__file__).resolve()),
        artifact_path="source"
    )

    mlflow.log_artifact(
        str(ROOT / "requirements.txt"),
        artifact_path="source"
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_val,
        predictions,
        ax=ax
    )

    mlflow.log_figure(
        fig,
        "validation_confusion_matrix.png"
    )

    plt.close(fig)

    signature = infer_signature(
        X_train,
        model.predict(X_train)
    )

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train.head(2),
        signature=signature,
        pip_requirements=(
            ROOT / "requirements.txt"
        ).read_text().splitlines()
    )

    print("RUN_ID:", run.info.run_id)
    print("MODEL_URI:", model_info.model_uri)
    print("VAL_ACCURACY:", accuracy)
    print("VAL_F1_MACRO:", f1_macro)

    if args.register:
        version = mlflow.register_model(
            model_info.model_uri,
            "heart-disease-classifier"
        )

        print("REGISTERED_VERSION:", version.version)

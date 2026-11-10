
import json
import logging
import os
import time
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(ROOT / "exported_model")
    )
)

app = FastAPI(
    title="Heart Disease Classifier",
    version="1.0.0"
)

model = mlflow.sklearn.load_model(str(MODEL_PATH))

metadata = json.loads(
    (MODEL_PATH / "metadata.json").read_text(
        encoding="utf-8"
    )
)

logger = logging.getLogger("uvicorn.error")


class PredictRequest(BaseModel):
    instances: list[dict[str, float | None]] = Field(
        min_length=1,
        max_length=100
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": metadata["version"]
    }


@app.get("/metadata")
def get_metadata():
    return metadata


@app.post("/predict")
def predict(body: PredictRequest):
    start = time.perf_counter()

    required = set(metadata["features"])

    for row in body.instances:
        if set(row.keys()) != required:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Invalid feature set",
                    "required_features": metadata["features"]
                }
            )

    X = pd.DataFrame(body.instances)
    X = X[metadata["features"]]
    X = X.astype(float)

    values = X.to_numpy(dtype=float)

    if np.isinf(values).any():
        raise HTTPException(
            status_code=422,
            detail="Feature values must be finite or null"
        )

    predictions = model.predict(X).astype(int).tolist()

    elapsed = (time.perf_counter() - start) * 1000

    logger.info(
        "model_version=%s rows=%d latency_ms=%.3f",
        metadata["version"],
        len(X),
        elapsed
    )

    return {
        "predictions": predictions,
        "class_names": [
            metadata["classes"][p]
            for p in predictions
        ],
        "model_version": metadata["version"],
        "latency_ms": round(elapsed, 3)
    }

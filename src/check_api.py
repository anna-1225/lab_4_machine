
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

request_body = (
    ROOT / "sample_request.json"
).read_bytes()

request = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=request_body,
    headers={
        "Content-Type": "application/json"
    }
)

with urllib.request.urlopen(
    request,
    timeout=30
) as response:
    result = json.load(response)

expected = json.loads(
    (ROOT / "expected_predictions.json").read_text(
        encoding="utf-8"
    )
)

metadata = json.loads(
    (ROOT / "exported_model/metadata.json").read_text(
        encoding="utf-8"
    )
)

assert result["predictions"] == expected, (
    result["predictions"],
    expected
)

assert str(result["model_version"]) == str(
    metadata["version"]
)

print("PASS: API predictions match exported model")
print("Model version:", result["model_version"])
print("Predictions:", result["predictions"])

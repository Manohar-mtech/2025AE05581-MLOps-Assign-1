"""Inference helpers shared by the API and offline scripts."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH

LABELS = {0: "No heart disease", 1: "Heart disease"}


def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m src.train`.")
    return joblib.load(path)


def to_frame(records: list) -> pd.DataFrame:
    """Build a model-ready frame: fixed column order, float dtype, None -> NaN."""
    return pd.DataFrame(records, columns=FEATURES).astype(float)


def predict_records(model, records: list) -> list:
    proba = model.predict_proba(to_frame(records))[:, 1]
    out = []
    for p in proba:
        label = int(p >= 0.5)
        out.append({
            "prediction": label,
            "label": LABELS[label],
            "confidence": round(float(p if label else 1 - p), 4),
            "probability_disease": round(float(p), 4),
        })
    return out


if __name__ == "__main__":  # python -m src.predict sample_input.json
    payload = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(predict_records(load_model(), [payload]), indent=2))

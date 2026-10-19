"""Shared fixtures: a small synthetic dataset and a quickly-trained model."""
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.modeling import MODEL_SPECS, make_pipeline


def make_synthetic(n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 75, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(95, 180, n),
        "chol": rng.integers(150, 350, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(100, 200, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }).astype(float)
    score = (df["age"] - 50) / 10 + df["oldpeak"] + df["exang"] + rng.normal(0, 1, n)
    df[TARGET] = (score > score.median()).astype(int)
    df.loc[df.index[:5], "ca"] = np.nan
    df.loc[df.index[5:8], "thal"] = np.nan
    return df


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic()


@pytest.fixture(scope="session")
def fitted_model(synthetic_df):
    est, _ = MODEL_SPECS["logistic_regression"]
    return make_pipeline(est).fit(synthetic_df[FEATURES], synthetic_df[TARGET])


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, fitted_model):
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(fitted_model, path)
    return path


@pytest.fixture()
def sample_payload():
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6}

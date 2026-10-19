"""Data loading, cleaning and splitting."""
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (CLEAN_PATH, FEATURES, RANDOM_STATE, RAW_COLUMNS,
                        RAW_PATH, TARGET, TEST_SIZE)

log = logging.getLogger(__name__)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks missing values."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first.")
    return pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, de-duplicate and binarise the target.

    Missing values (ca, thal) are deliberately kept as NaN here and imputed
    inside the sklearn Pipeline, so imputation statistics are learned from
    training folds only (no leakage) and the API reuses the same logic.
    """
    missing_cols = set(RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")
    out = df.drop_duplicates().reset_index(drop=True)
    out[TARGET] = (out["num"] > 0).astype(int)  # 0 = healthy, 1..4 = disease
    out = out.drop(columns="num")
    out[FEATURES] = out[FEATURES].astype(float)
    return out


def build_clean_dataset(raw_path: Path = RAW_PATH, out_path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = clean(load_raw(raw_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved cleaned dataset %s to %s", df.shape, out_path)
    return df


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return build_clean_dataset()
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


def train_test(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE):
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = build_clean_dataset()
    print(d.describe(include="all").T)

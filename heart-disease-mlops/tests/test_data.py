import pandas as pd
import pytest

from src.config import FEATURES, RAW_COLUMNS, TARGET
from src.data import clean, load_raw, split_xy, train_test


def test_load_raw_has_expected_shape():
    df = load_raw()
    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 303


def test_raw_missing_values_parsed_as_nan():
    df = load_raw()
    assert df["ca"].isna().sum() == 4
    assert df["thal"].isna().sum() == 2


def test_clean_binarises_target():
    cleaned = clean(load_raw())
    assert set(cleaned[TARGET].unique()) == {0, 1}
    assert "num" not in cleaned.columns


def test_clean_casts_features_to_float():
    cleaned = clean(load_raw())
    assert all(cleaned[c].dtype == float for c in FEATURES)


def test_clean_drops_duplicates():
    raw = load_raw()
    doubled = pd.concat([raw, raw], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw))


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError):
        clean(load_raw().drop(columns=["chol"]))


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.data")


def test_split_is_stratified_and_disjoint():
    df = clean(load_raw())
    X_tr, X_te, y_tr, y_te = train_test(df, test_size=0.2, seed=1)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(df)
    assert list(split_xy(df)[0].columns) == FEATURES

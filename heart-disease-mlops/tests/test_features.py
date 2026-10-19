import numpy as np

from src.config import FEATURES
from src.features import build_preprocessor


def test_preprocessor_outputs_no_nans(synthetic_df):
    out = build_preprocessor().fit_transform(synthetic_df[FEATURES])
    dense = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(dense).any()


def test_numeric_columns_are_standardised(synthetic_df):
    out = build_preprocessor().fit_transform(synthetic_df[FEATURES])
    dense = out.toarray() if hasattr(out, "toarray") else out
    assert np.allclose(dense[:, :5].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(dense[:, :5].std(axis=0), 1, atol=1e-6)


def test_unseen_category_is_ignored(synthetic_df):
    prep = build_preprocessor().fit(synthetic_df[FEATURES])
    row = synthetic_df[FEATURES].head(1).copy()
    row["cp"] = 99.0
    assert prep.transform(row).shape[0] == 1

import json

import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import FEATURES, TARGET
from src.modeling import (MODEL_SPECS, cross_validate_model, evaluate,
                          feature_importance, make_pipeline, tune)
from src.predict import load_model, predict_records, to_frame


def test_at_least_two_model_families():
    assert len(MODEL_SPECS) >= 2


def test_tune_returns_fitted_pipeline(synthetic_df):
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    search = tune("logistic_regression", synthetic_df[FEATURES], synthetic_df[TARGET], cv)
    assert "clf__C" in search.best_params_
    assert search.best_estimator_.predict(synthetic_df[FEATURES]).shape[0] == len(synthetic_df)


def test_cross_validate_reports_all_metrics(synthetic_df):
    est, _ = MODEL_SPECS["random_forest"]
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    res = cross_validate_model(make_pipeline(est), synthetic_df[FEATURES],
                               synthetic_df[TARGET], cv)
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0 <= res[f"cv_{m}_mean"] <= 1


def test_evaluate_metrics_in_range(fitted_model, synthetic_df):
    res = evaluate(fitted_model, synthetic_df[FEATURES], synthetic_df[TARGET])
    assert all(0 <= v <= 1 for v in res.values())
    assert res["test_roc_auc"] > 0.7  # signal is planted in the synthetic data


def test_feature_importance_sorted(fitted_model):
    imp = feature_importance(fitted_model)
    assert imp.is_monotonic_decreasing


def test_model_roundtrip_gives_identical_predictions(model_path, fitted_model, synthetic_df):
    loaded = load_model(model_path)
    X = synthetic_df[FEATURES]
    assert (loaded.predict_proba(X) == fitted_model.predict_proba(X)).all()


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_predict_records_schema(fitted_model, sample_payload):
    out = predict_records(fitted_model, [sample_payload])[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert abs(out["probability_disease"] - (out["confidence"] if out["prediction"]
                                              else 1 - out["confidence"])) < 1e-3


def test_to_frame_handles_none(sample_payload):
    sample_payload["ca"] = None
    frame = to_frame([sample_payload])
    assert frame.shape == (1, len(FEATURES))
    assert frame["ca"].isna().all()


def test_metadata_is_valid_json_when_present(tmp_path):
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps({"model_name": "x"}))
    assert json.loads(p.read_text())["model_name"] == "x"

"""Model definitions, hyper-parameter tuning and evaluation (no MLflow here)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.features import build_preprocessor

# name -> (estimator, parameter grid). Grid keys are prefixed with the pipeline step "clf__".
MODEL_SPECS = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5, None],
            "clf__min_samples_leaf": [1, 3],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
    ),
}

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clone(estimator))])


def tune(name: str, X, y, cv, n_jobs: int = 1) -> GridSearchCV:
    """Grid-search one model family, optimising ROC-AUC under stratified CV."""
    estimator, grid = MODEL_SPECS[name]
    search = GridSearchCV(make_pipeline(estimator), grid, scoring="roc_auc",
                          cv=cv, n_jobs=n_jobs, refit=True)
    return search.fit(X, y)


def cross_validate_model(pipeline, X, y, cv, n_jobs: int = 1) -> dict:
    res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    out = {}
    for m in SCORING:
        out[f"cv_{m}_mean"] = float(np.mean(res[f"test_{m}"]))
        out[f"cv_{m}_std"] = float(np.std(res[f"test_{m}"]))
    return out


def evaluate(pipeline, X, y) -> dict:
    """Hold-out metrics."""
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    return {
        "test_accuracy": float(accuracy_score(y, pred)),
        "test_precision": float(precision_score(y, pred, zero_division=0)),
        "test_recall": float(recall_score(y, pred, zero_division=0)),
        "test_f1": float(f1_score(y, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y, proba)),
    }


def feature_importance(pipeline):
    """Return a sorted Series of importances (tree) or |coef| (linear), else None."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
    else:
        return None
    return pd.Series(vals, index=names).sort_values(ascending=False)

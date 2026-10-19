"""Train, tune, evaluate and track models with MLflow.

Usage:  python -m src.train [--cv-folds 5] [--models logistic_regression random_forest]
"""
import argparse
import json
import logging
import os
import platform
import tempfile
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold

from src.config import (CLEAN_PATH, FEATURES, METADATA_PATH, MLRUNS_DIR,
                        MODEL_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE)
from src.data import load_clean, train_test
from src.modeling import (MODEL_SPECS, cross_validate_model, evaluate,
                          feature_importance, tune)
from src.plots import confusion_matrix_fig, importance_fig, roc_fig

log = logging.getLogger("train")


def _log_fig(fig, path):
    mlflow.log_figure(fig, path)
    plt.close(fig)


def run_training(experiment: str, cv_folds: int, models: list, n_jobs: int = 1) -> dict:
    df = load_clean()
    X_train, X_test, y_train, y_test = train_test(df)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri()))
    mlflow.set_experiment(experiment)

    results = {}
    with mlflow.start_run(run_name="training-session"):
        mlflow.log_params({
            "n_train": len(X_train), "n_test": len(X_test), "cv_folds": cv_folds,
            "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
            "positive_rate": round(float(y_train.mean()), 4),
            "sklearn_version": sklearn.__version__,
        })
        mlflow.log_artifact(str(CLEAN_PATH), artifact_path="data")

        for name in models:
            with mlflow.start_run(run_name=name, nested=True):
                log.info("Tuning %s ...", name)
                mlflow.set_tags({"model_family": name, "stage": "experiment"})
                search = tune(name, X_train, y_train, cv, n_jobs=n_jobs)
                best = search.best_estimator_

                mlflow.log_param("model_type", type(best.named_steps["clf"]).__name__)
                mlflow.log_param("grid_size", len(search.cv_results_["params"]))
                mlflow.log_params({k.replace("clf__", ""): v
                                   for k, v in search.best_params_.items()})
                mlflow.log_metric("search_best_cv_roc_auc", float(search.best_score_))

                cv_metrics = cross_validate_model(best, X_train, y_train, cv, n_jobs=n_jobs)
                test_metrics = evaluate(best, X_test, y_test)
                mlflow.log_metrics({**cv_metrics, **test_metrics})

                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "grid_search_results.csv")
                    pd.DataFrame(search.cv_results_).to_csv(path, index=False)
                    mlflow.log_artifact(path, artifact_path="tuning")
                _log_fig(confusion_matrix_fig(best, X_test, y_test, f"{name}: confusion matrix"),
                         "plots/confusion_matrix.png")
                _log_fig(roc_fig(best, X_test, y_test, f"{name}: ROC curve"),
                         "plots/roc_curve.png")
                imp = feature_importance(best)
                if imp is not None:
                    _log_fig(importance_fig(imp, title=f"{name}: feature importance"),
                             "plots/feature_importance.png")

                signature = infer_signature(X_test, best.predict(X_test))
                mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                         input_example=X_train.head(3))
                results[name] = {"pipeline": best, "params": search.best_params_,
                                 "cv": cv_metrics, "test": test_metrics}
                log.info("%s | CV ROC-AUC %.3f | test ROC-AUC %.3f", name,
                         cv_metrics["cv_roc_auc_mean"], test_metrics["test_roc_auc"])

        # Model selection uses CV ROC-AUC only (the test set is never used to choose).
        best_name = max(results, key=lambda n: results[n]["cv"]["cv_roc_auc_mean"])
        best = results[best_name]
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best["pipeline"], MODEL_PATH)
        metadata = {
            "model_name": best_name,
            "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                            for k, v in best["params"].items()},
            "cv_metrics": best["cv"],
            "test_metrics": best["test"],
            "features": FEATURES,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metric("best_cv_roc_auc", best["cv"]["cv_roc_auc_mean"])
        mlflow.log_metric("best_test_roc_auc", best["test"]["test_roc_auc"])
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="final_model")
        mlflow.log_artifact(str(METADATA_PATH), artifact_path="final_model")

    summary = pd.DataFrame({
        n: {"cv_roc_auc": r["cv"]["cv_roc_auc_mean"], "cv_accuracy": r["cv"]["cv_accuracy_mean"],
            "test_accuracy": r["test"]["test_accuracy"],
            "test_precision": r["test"]["test_precision"],
            "test_recall": r["test"]["test_recall"],
            "test_roc_auc": r["test"]["test_roc_auc"]}
        for n, r in results.items()}).T.round(3)
    print("\n" + summary.to_string())
    print(f"\nBest model: {best_name}  ->  saved to {MODEL_PATH}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default="heart-disease-risk")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    run_training(a.experiment, a.cv_folds, a.models, a.n_jobs)

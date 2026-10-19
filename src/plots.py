"""Matplotlib figure helpers used for MLflow artifacts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402


def confusion_matrix_fig(pipeline, X, y, title="Confusion matrix (test set)"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X, y, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def roc_fig(pipeline, X, y, title="ROC curve (test set)"):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X, y, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def importance_fig(series, top=15, title="Top feature importances"):
    s = series.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(s.index, s.values, color="#3b6ea5")
    ax.set_title(title)
    fig.tight_layout()
    return fig

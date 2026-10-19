"""Exploratory data analysis: writes figures to reports/figures/."""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import CATEGORICAL_FEATURES, FIG_DIR, NUMERIC_FEATURES, TARGET  # noqa: E402
from src.data import build_clean_dataset  # noqa: E402

log = logging.getLogger("eda")
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4c9f70", 1: "#c0392b"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    log.info("wrote %s", FIG_DIR / name)


def run():
    df = build_clean_dataset()
    print("Shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])
    print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3))

    # class balance
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.countplot(x=TARGET, data=df, hue=TARGET, palette=PALETTE, legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_title("Class balance")
    _save(fig, "class_balance.png")

    # missing values
    miss = df.isna().sum()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    miss.plot.bar(ax=ax, color="#7f8c8d")
    ax.set_title("Missing values per column")
    _save(fig, "missing_values.png")

    # numeric histograms
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, palette=PALETTE, kde=True,
                     element="step", ax=ax)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric feature distributions by target")
    _save(fig, "histograms_numeric.png")

    # boxplots
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=PALETTE,
                    legend=False, ax=ax)
    fig.suptitle("Numeric features vs target")
    _save(fig, "boxplots_numeric.png")

    # categorical vs target
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=TARGET, palette=PALETTE, ax=ax)
        ax.legend(title="target", fontsize=8)
    fig.suptitle("Categorical features by target")
    _save(fig, "categorical_vs_target.png")

    # correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation heatmap")
    _save(fig, "correlation_heatmap.png")

    print("\nCorrelation with target:\n",
          df.corr()[TARGET].drop(TARGET).sort_values(ascending=False).round(3))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    run()

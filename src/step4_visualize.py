"""Comparison plots across the three feature sets."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES_DIR, OUTPUTS_DIR

FEATURE_SETS = ["raw", "pca", "clusters"]
COLORS = {"raw": "#4C72B0", "pca": "#DD8452", "clusters": "#55A868"}


def grouped_bar(df, metric, ylabel, filename, lower_is_better=False):
    models = sorted(df["model"].unique())
    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, fs in enumerate(FEATURE_SETS):
        vals = [
            df[(df["model"] == m) & (df["feature_set"] == fs)][metric].mean()
            for m in models
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=fs, color=COLORS[fs])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by model and feature set")
    ax.legend(title="feature set")
    if lower_is_better:
        ax.set_yscale("log")
    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    path = OUTPUTS_DIR / "final_comparison.csv"
    if not path.exists():
        print(f"missing {path} -- run step3_evaluate.py first")
        return

    df = pd.read_csv(path)
    grouped_bar(df, "roc_auc", "ROC-AUC", "roc_auc_comparison.png")
    grouped_bar(df, "pr_auc", "PR-AUC", "pr_auc_comparison.png")
    grouped_bar(df, "accuracy", "Accuracy", "accuracy_comparison.png")
    grouped_bar(df, "f1", "F1 score", "f1_comparison.png")
    grouped_bar(df, "train_seconds", "Training time (s)", "train_time_comparison.png", lower_is_better=True)
    grouped_bar(df, "inference_seconds", "Inference time (s)", "inference_time_comparison.png", lower_is_better=True)


if __name__ == "__main__":
    main()

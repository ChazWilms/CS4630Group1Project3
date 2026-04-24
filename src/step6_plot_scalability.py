"""Scalability plots from outputs/scalability.csv.

- Log-log train_seconds vs n_train, per model, with fitted power-law exponent.
- Semilog ROC-AUC vs n_train, per model, to show plateau behavior.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES_DIR, OUTPUTS_DIR

MODEL_ORDER = [
    "linear_svm",
    "rbf_svm",
    "knn",
    "decision_tree",
    "random_forest",
    "xgboost",
]
COLORS = {
    "linear_svm": "#4C72B0",
    "rbf_svm": "#DD8452",
    "knn": "#55A868",
    "decision_tree": "#C44E52",
    "random_forest": "#8172B2",
    "xgboost": "#937860",
}


def fit_exponent(n, t):
    # log10(t) = a + b*log10(n); b is the scaling exponent
    b, _ = np.polyfit(np.log10(n), np.log10(t), 1)
    return b


def plot_time(df, col, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in MODEL_ORDER:
        sub = df[df["model"] == name].sort_values("n_train")
        n = sub["n_train"].values
        t = sub[col].values
        if len(n) >= 4 and (t > 0).all():
            b = fit_exponent(n, t)
            label = f"{name} (~n^{b:.2f})"
        else:
            label = name
        ax.plot(n, t, "o-", color=COLORS[name], label=label, linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} scaling (raw features, fixed best params)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_roc_auc(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in MODEL_ORDER:
        sub = df[df["model"] == name].sort_values("n_train")
        ax.plot(sub["n_train"], sub["roc_auc"], "o-", color=COLORS[name],
                label=name, linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (n)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Accuracy vs training set size (raw features, fixed best params)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "scalability_roc_auc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    path = OUTPUTS_DIR / "scalability.csv"
    if not path.exists():
        print(f"missing {path} -- run step5_scalability.py first")
        return
    df = pd.read_csv(path)
    plot_time(df, "train_seconds", "Training time (s)", "scalability_train_time.png")
    plot_time(df, "inference_seconds", "Inference time (s)", "scalability_inference_time.png")
    plot_roc_auc(df)


if __name__ == "__main__":
    main()

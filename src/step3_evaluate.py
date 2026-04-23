"""Combine the three results CSVs into one comparison table."""

import pandas as pd

from common import OUTPUTS_DIR


def main():
    frames = []
    for fs in ("raw", "pca", "clusters"):
        path = OUTPUTS_DIR / f"results_{fs}.csv"
        if not path.exists():
            print(f"missing {path} -- run step2{'abc'['raw pca clusters'.split().index(fs)]}_* first")
            continue
        frames.append(pd.read_csv(path))

    if not frames:
        print("No results to combine.")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df[[
        "feature_set", "model",
        "accuracy", "f1", "roc_auc", "pr_auc",
        "train_seconds", "inference_seconds",
        "cv_best_roc_auc", "best_params",
    ]]

    out = OUTPUTS_DIR / "final_comparison.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

    pivot = df.pivot(index="model", columns="feature_set", values="roc_auc")
    print("\nROC-AUC by model x feature set:")
    print(pivot.round(4))


if __name__ == "__main__":
    main()

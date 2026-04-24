"""Scalability experiment: vary training set size with fixed best hyperparameters.

For each (model, size), fit ONCE with the hyperparameters already chosen via
GridSearchCV on the full 160k. Records train_seconds, inference_seconds, and
test ROC-AUC so we can plot scaling curves.

RBF-SVM caps at 40k rows; beyond that one fit takes tens of minutes per point.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import OUTPUTS_DIR, SPLITS_DIR, compute_metrics, get_scores, timed

FEATURE_SET = "raw"
SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
RBF_MAX = 40_000

# Best params from final_comparison.csv (raw feature set, GridSearchCV on 160k).
BEST_PARAMS = {
    "linear_svm": {"C": 10.0},
    "rbf_svm": {"C": 10.0, "gamma": 0.01},
    "knn": {"n_neighbors": 31},
    "decision_tree": {"max_depth": 12, "min_samples_leaf": 50},
    "random_forest": {"max_depth": None, "min_samples_leaf": 5, "n_estimators": 200},
    "xgboost": {"learning_rate": 0.05, "max_depth": 8, "n_estimators": 400},
}


def make_model(name):
    p = BEST_PARAMS[name]
    if name == "linear_svm":
        return LinearSVC(dual="auto", max_iter=5000, random_state=42, **p)
    if name == "rbf_svm":
        return SVC(kernel="rbf", probability=True, random_state=42, cache_size=4000, **p)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1, **p)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=42, **p)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=42, **p)
    if name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            **p,
        )
    raise ValueError(f"Unknown model: {name}")


def stratified_subsample(X, y, size, rng):
    if size >= len(y):
        return X, y
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    frac = size / len(y)
    pos_sel = rng.choice(pos, size=int(round(len(pos) * frac)), replace=False)
    neg_sel = rng.choice(neg, size=size - len(pos_sel), replace=False)
    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return X[idx], y[idx]


def main():
    X_train = np.load(SPLITS_DIR / f"X_{FEATURE_SET}_train.npy")
    X_test = np.load(SPLITS_DIR / f"X_{FEATURE_SET}_test.npy")
    y_train = np.load(SPLITS_DIR / "y_train.npy")
    y_test = np.load(SPLITS_DIR / "y_test.npy")

    print(f"Feature set: {FEATURE_SET}")
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    rng = np.random.default_rng(42)
    rows = []
    for name in BEST_PARAMS:
        for size in SIZES:
            if name == "rbf_svm" and size > RBF_MAX:
                print(f"\n[skip] {name} @ n={size:,} (above RBF cap {RBF_MAX:,})")
                continue

            X_fit, y_fit = stratified_subsample(X_train, y_train, size, rng)
            model = make_model(name)

            print(f"\n--- {name} @ n={len(y_fit):,} ---")
            with timed(f"{name} fit") as t_fit:
                model.fit(X_fit, y_fit)
            with timed(f"{name} predict") as t_predict:
                y_pred = model.predict(X_test)
                y_proba = get_scores(model, X_test)

            metrics = compute_metrics(y_test, y_pred, y_proba)
            rows.append({
                "model": name,
                "n_train": len(y_fit),
                "train_seconds": t_fit["seconds"],
                "inference_seconds": t_predict["seconds"],
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "pr_auc": metrics["pr_auc"],
            })
            print(f"  ROC-AUC={metrics['roc_auc']:.4f}  train={t_fit['seconds']:.2f}s")

    out_path = OUTPUTS_DIR / "scalability.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

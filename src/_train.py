"""Shared training loop used by step2a/step2b/step2c.

Loads a named feature set from splits/, runs GridSearchCV for each model,
records training time, inference time, best params, and test metrics.
Writes one CSV to outputs/results_<feature_set>.csv.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from common import (
    OUTPUTS_DIR,
    RBF_FULL_PARAMS,
    SPLITS_DIR,
    compute_metrics,
    get_models,
    get_scores,
    timed,
)

CV_FOLDS = 3

# RBF-SVM scales O(n^2) and is infeasible on 160k rows.
# Subsample its training set only (stratified by class) to keep tuning tractable.
# Bypassed when rbf_full=True (step2e uses pre-selected params + full data).
RBF_TRAIN_CAP = 20_000


def _maybe_subsample(name, X, y, rng):
    if name != "rbf_svm" or len(y) <= RBF_TRAIN_CAP:
        return X, y
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    frac = RBF_TRAIN_CAP / len(y)
    pos_sel = rng.choice(pos, size=int(len(pos) * frac), replace=False)
    neg_sel = rng.choice(neg, size=RBF_TRAIN_CAP - len(pos_sel), replace=False)
    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    print(f"  [rbf_svm] subsampled training set: {len(y):,} -> {len(idx):,}")
    return X[idx], y[idx]


def run(feature_set, only_models=None, append=False, rbf_full=False, replaces=None):
    """Train models on a feature set and write results CSV.

    only_models : iterable of model names to include (None = all).
    append      : merge into existing results_<feature_set>.csv instead of overwriting.
                  Existing rows for any model in `only_models` or `replaces` are dropped.
    rbf_full    : use pre-selected single-cell RBF grid AND skip the 20k subsample cap.
    replaces    : extra model names whose existing rows should be dropped on append
                  (e.g. xgboost run with replaces=["gradient_boosting"]).
    """
    X_train = np.load(SPLITS_DIR / f"X_{feature_set}_train.npy")
    X_test = np.load(SPLITS_DIR / f"X_{feature_set}_test.npy")
    y_train = np.load(SPLITS_DIR / "y_train.npy")
    y_test = np.load(SPLITS_DIR / "y_test.npy")

    print(f"=== Feature set: {feature_set} ===")
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    if rbf_full:
        print(f"[rbf_full] pre-selected params: {RBF_FULL_PARAMS[feature_set]}")

    rbf_params = RBF_FULL_PARAMS[feature_set] if rbf_full else None
    models = get_models(rbf_full_params=rbf_params)

    rng = np.random.default_rng(42)
    rows = []
    for name, (estimator, grid) in models.items():
        if only_models is not None and name not in only_models:
            continue

        print(f"\n--- {name} ---")
        if rbf_full and name == "rbf_svm":
            X_fit, y_fit = X_train, y_train
            print(f"  [rbf_svm] training on full {len(y_fit):,} rows")
        else:
            X_fit, y_fit = _maybe_subsample(name, X_train, y_train, rng)
        gs = GridSearchCV(estimator, grid, cv=CV_FOLDS, scoring="roc_auc", n_jobs=1)

        with timed(f"{name} fit") as t_fit:
            gs.fit(X_fit, y_fit)

        best = gs.best_estimator_
        with timed(f"{name} predict") as t_predict:
            y_pred = best.predict(X_test)
            y_proba = get_scores(best, X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        rows.append({
            "feature_set": feature_set,
            "model": name,
            "n_train": len(y_fit),
            "train_seconds": t_fit["seconds"],
            "inference_seconds": t_predict["seconds"],
            "best_params": str(gs.best_params_),
            "cv_best_roc_auc": gs.best_score_,
            **metrics,
        })
        print(f"best={gs.best_params_}  test ROC-AUC={metrics['roc_auc']:.4f}")

    new_df = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / f"results_{feature_set}.csv"

    if append and out_path.exists():
        existing = pd.read_csv(out_path)
        drop_names = set(new_df["model"].unique()) | set(replaces or [])
        existing = existing[~existing["model"].isin(drop_names)]
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged.to_csv(out_path, index=False)
        print(f"\nAppended {len(new_df)} rows to {out_path} (dropped: {sorted(drop_names)})")
    else:
        new_df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

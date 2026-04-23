"""Shared training loop used by step2a/step2b/step2c.

Loads a named feature set from splits/, runs GridSearchCV for each model,
records training time, inference time, best params, and test metrics.
Writes one CSV to outputs/results_<feature_set>.csv.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from common import OUTPUTS_DIR, SPLITS_DIR, compute_metrics, get_models, get_scores, timed

CV_FOLDS = 3


def run(feature_set):
    X_train = np.load(SPLITS_DIR / f"X_{feature_set}_train.npy")
    X_test = np.load(SPLITS_DIR / f"X_{feature_set}_test.npy")
    y_train = np.load(SPLITS_DIR / "y_train.npy")
    y_test = np.load(SPLITS_DIR / "y_test.npy")

    print(f"=== Feature set: {feature_set} ===")
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    rows = []
    for name, (estimator, grid) in get_models().items():
        print(f"\n--- {name} ---")
        gs = GridSearchCV(estimator, grid, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1)

        with timed(f"{name} fit") as t_fit:
            gs.fit(X_train, y_train)

        best = gs.best_estimator_
        with timed(f"{name} predict") as t_predict:
            y_pred = best.predict(X_test)
            y_proba = get_scores(best, X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        rows.append({
            "feature_set": feature_set,
            "model": name,
            "train_seconds": t_fit["seconds"],
            "inference_seconds": t_predict["seconds"],
            "best_params": str(gs.best_params_),
            "cv_best_roc_auc": gs.best_score_,
            **metrics,
        })
        print(f"best={gs.best_params_}  test ROC-AUC={metrics['roc_auc']:.4f}")

    out_path = OUTPUTS_DIR / f"results_{feature_set}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

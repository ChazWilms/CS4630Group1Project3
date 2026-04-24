"""Shared helpers: paths, timing, metrics, model registry."""

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# -- Paths --------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = ROOT_DIR / "splits"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = ROOT_DIR / "figures"

for d in (SPLITS_DIR, OUTPUTS_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True)


# -- Timing -------------------------------------------------------------------

@contextmanager
def timed(label=""):
    start = time.perf_counter()
    result = {"seconds": None}
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        if label:
            print(f"[{label}] {result['seconds']:.2f}s")


# -- Metrics ------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


# -- Model registry -----------------------------------------------------------
#
# Each entry: name -> (estimator, param_grid)
# Small 3x3 grids keep cross-validation tractable on a 160k training set.
# Models marked `scale_sensitive=True` in NEEDS_SCALING should receive
# pre-scaled features (X_scaled.csv).

NEEDS_SCALING = {"linear_svm", "rbf_svm", "knn"}

# Pre-selected RBF-SVM params from 20k pilot run, keyed by feature set.
# Used by step2e_rbf_full.py to train on the full 160k without a 27-hour grid.
RBF_FULL_PARAMS = {
    "raw": {"C": 10.0, "gamma": 0.01},
    "pca": {"C": 1.0, "gamma": "scale"},
    "clusters": {"C": 10.0, "gamma": 0.01},
}


def get_models(rbf_full_params=None):
    """Return model registry.

    If rbf_full_params is provided (dict like {"C": 10.0, "gamma": 0.01}),
    the RBF-SVM grid collapses to a single cell so we can train on full data.
    """
    if rbf_full_params is not None:
        rbf_grid = {k: [v] for k, v in rbf_full_params.items()}
    else:
        rbf_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]}

    models = {
        "linear_svm": (
            LinearSVC(dual="auto", max_iter=5000, random_state=42),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "rbf_svm": (
            SVC(kernel="rbf", probability=True, random_state=42, cache_size=4000),
            rbf_grid,
        ),
        "knn": (
            KNeighborsClassifier(n_jobs=-1),
            {"n_neighbors": [5, 15, 31]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [6, 12, None], "min_samples_leaf": [1, 10, 50]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
            {"max_depth": [8, 16, None], "min_samples_leaf": [1, 5, 20]},
        ),
    }
    if HAS_XGBOOST:
        models["xgboost"] = (
            XGBClassifier(
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
            },
        )
    else:
        models["gradient_boosting"] = (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        )
    return models


# -- Probability helper -------------------------------------------------------
#
# LinearSVC has no predict_proba; use decision_function and min-max to [0,1]
# so AUC metrics still work on a consistent scale.

def get_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    scores = estimator.decision_function(X)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - lo) / (hi - lo)

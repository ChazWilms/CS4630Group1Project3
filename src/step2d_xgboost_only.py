"""Train XGBoost only on raw, pca, and clusters feature sets.

Appends xgboost rows to outputs/results_<feature_set>.csv and drops any
existing gradient_boosting rows, since XGBoost replaces it per the project spec.
"""

from _train import run

if __name__ == "__main__":
    for feature_set in ("raw", "pca", "clusters"):
        run(
            feature_set,
            only_models=["xgboost"],
            append=True,
            replaces=["gradient_boosting"],
        )

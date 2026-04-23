"""Train RBF-SVM on the full 160k using pre-selected params from the 20k pilot.

Uses a single-cell grid (no CV search) per feature set so the whole run
finishes in hours instead of days. Appends / replaces the rbf_svm row in
outputs/results_<feature_set>.csv.
"""

from _train import run

if __name__ == "__main__":
    for feature_set in ("raw", "pca", "clusters"):
        run(
            feature_set,
            only_models=["rbf_svm"],
            append=True,
            rbf_full=True,
        )

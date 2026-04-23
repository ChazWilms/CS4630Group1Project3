"""Stratified 80/20 train/test split on the Project 2 scaled subsample."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_DIR, SPLITS_DIR

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    X = pd.read_csv(DATA_DIR / "X_scaled.csv").to_numpy()
    y = pd.read_csv(DATA_DIR / "y.csv").iloc[:, 0].to_numpy().astype(int)

    X_pca = pd.read_csv(DATA_DIR / "X_pca_10.csv").to_numpy()
    clusters = pd.read_csv(DATA_DIR / "cluster_labels.csv").iloc[:, 0].to_numpy().astype(int)

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    np.save(SPLITS_DIR / "idx_train.npy", idx_train)
    np.save(SPLITS_DIR / "idx_test.npy", idx_test)

    np.save(SPLITS_DIR / "X_raw_train.npy", X[idx_train])
    np.save(SPLITS_DIR / "X_raw_test.npy", X[idx_test])
    np.save(SPLITS_DIR / "X_pca_train.npy", X_pca[idx_train])
    np.save(SPLITS_DIR / "X_pca_test.npy", X_pca[idx_test])

    X_clu_train = np.hstack([X[idx_train], clusters[idx_train].reshape(-1, 1)])
    X_clu_test = np.hstack([X[idx_test], clusters[idx_test].reshape(-1, 1)])
    np.save(SPLITS_DIR / "X_clusters_train.npy", X_clu_train)
    np.save(SPLITS_DIR / "X_clusters_test.npy", X_clu_test)

    np.save(SPLITS_DIR / "y_train.npy", y[idx_train])
    np.save(SPLITS_DIR / "y_test.npy", y[idx_test])

    print(f"Train size: {len(idx_train):,}  Test size: {len(idx_test):,}")
    print(f"Train positives: {y[idx_train].mean():.4f}  Test positives: {y[idx_test].mean():.4f}")
    print(f"Splits saved to {SPLITS_DIR}")


if __name__ == "__main__":
    main()

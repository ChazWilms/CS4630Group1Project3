"""Load 200k rows from HIGGS, save unscaled + scaled features, labels,
PCA-10 projection, and k-Means cluster IDs. Self-contained — regenerates
all inputs the training scripts need.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import DATA_DIR, timed

N_ROWS = 200_000
N_COMPONENTS = 10
N_CLUSTERS = 2
RANDOM_STATE = 42


def main():
    cols = ["label"] + [f"feature_{i}" for i in range(28)]
    raw_path = DATA_DIR / "HIGGS.csv.gz"

    with timed("load HIGGS"):
        df = pd.read_csv(raw_path, names=cols, nrows=N_ROWS)

    X_unscaled = df.drop("label", axis=1).to_numpy()
    y = df["label"].to_numpy().astype(int)

    scaler = StandardScaler()
    with timed("standard scale"):
        X_scaled = scaler.fit_transform(X_unscaled)

    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    with timed(f"PCA to {N_COMPONENTS} components"):
        X_pca = pca.fit_transform(X_scaled)
    print(f"PCA cumulative variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    with timed(f"k-Means k={N_CLUSTERS} on PCA"):
        km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
        clusters = km.fit_predict(X_pca)

    pd.DataFrame(X_unscaled, columns=[f"feature_{i}" for i in range(28)]).to_csv(
        DATA_DIR / "X_unscaled.csv", index=False
    )
    pd.DataFrame(X_scaled, columns=[f"feature_{i}" for i in range(28)]).to_csv(
        DATA_DIR / "X_scaled.csv", index=False
    )
    pd.Series(y, name="label").to_csv(DATA_DIR / "y.csv", index=False)
    pd.DataFrame(X_pca, columns=[f"pc_{i}" for i in range(N_COMPONENTS)]).to_csv(
        DATA_DIR / "X_pca_10.csv", index=False
    )
    pd.Series(clusters, name="cluster").to_csv(DATA_DIR / "cluster_labels.csv", index=False)

    print(f"\nRows: {N_ROWS:,}   Positives: {y.mean():.4f}")
    print(f"Wrote X_unscaled.csv, X_scaled.csv, y.csv, X_pca_10.csv, cluster_labels.csv to {DATA_DIR}")


if __name__ == "__main__":
    main()

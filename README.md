# Project 3: Large-Scale Supervised Classification on the UCI HIGGS Dataset

**Course:** CS 4/5630 - Python for Computational and Data Sciences
**Instructor:** Dr. Arijit Khan
**Group 1:** Quang Minh Nguyen, Rachel Stevenson, Jack Handley, Isaac Avila, Vaughn Gugger, Chaz Wilms

---

## Project Overview

This project trains and compares six classic supervised classifiers on the UCI HIGGS dataset (binary classification: signal vs. background). It then integrates the unsupervised results from Project 2 by (A) using PCA-reduced features as input, and (B) adding k-Means cluster IDs as additional features, to measure how dimensionality reduction and clustering affect classification performance.

---

## Dataset

The raw data is not stored in this repository due to file size. Download from:

**[Download Raw Data (OneDrive)](https://falconbgsu-my.sharepoint.com/personal/cwilms_bgsu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcwilms%5Fbgsu%5Fedu%2FDocuments%2FCS4630%20Group%201%20%2D%20Data%2FProject%202&viewid=94f93628%2Da265%2D4f13%2Da339%2Dab773cd74794)**

Once downloaded, place the files in the repository as follows:

```text
data/
├── HIGGS.csv.gz          # Raw compressed dataset (11M rows, 28 features)
├── X_scaled.csv          # Scaled 200k subsample (from Project 2)
├── y.csv                 # Labels for the subsample (from Project 2)
├── X_pca_10.csv          # PCA-10 features (from Project 2)
└── cluster_labels.csv    # Best-k k-Means cluster IDs (from Project 2)
```

---

## Repository Structure

```text
CS4630Group1Project3/
├── data/                        # Raw + Project 2 artifacts
├── src/                         # Pipeline source code
├── splits/                      # Train/test splits (generated, not committed)
├── outputs/                     # Metrics CSVs
├── figures/                     # Plots for the report
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Python Version

Python 3.9 or higher is recommended.

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Pipeline

Run each step in order from the `src/` directory. All outputs are written to `outputs/` and `figures/`.

```bash
cd src
```

### Step 0 — Preprocess the raw data

```bash
python step0_preprocess.py
```

Reads `data/HIGGS.csv.gz`, subsamples 200k rows (stratified), scales features, and writes `data/X_scaled.csv`, `data/X_unscaled.csv`, and `data/y.csv`.

### Step 1 — Prepare train/test splits

```bash
python step1_prepare_splits.py
```

Performs a stratified split and saves NumPy arrays to `splits/` for each feature variant (raw, PCA-10, cluster-augmented).

### Step 2 — Train classifiers

Run all three scripts; each writes its own results CSV.

```bash
python step2a_train_raw.py       # 6 models on raw 28 features
python step2b_train_pca.py       # 6 models on PCA-10 features
python step2c_train_clusters.py  # 6 models on raw + cluster-ID feature
```

Each script runs GridSearchCV (3-fold CV) then re-fits with the best params on the full training set. Results are written to `outputs/results_raw.csv`, `outputs/results_pca.csv`, and `outputs/results_clusters.csv`.

> **Note on RBF-SVM:** The grid search runs on a 20k subsample due to quadratic complexity. To train on the full 160k using pre-selected params, run `step2e_rbf_full.py` after the above.

### Step 3 — Combine results

```bash
python step3_evaluate.py
```

Merges the three results CSVs into `outputs/final_comparison.csv` and prints an ROC-AUC pivot table.

### Step 4 — Generate comparison plots

```bash
python step4_visualize.py
```

Writes six bar-chart PNGs to `figures/` (accuracy, F1, ROC-AUC, PR-AUC, train time, inference time).

### Step 5 — Scalability experiment

```bash
python step5_scalability.py
```

Re-trains each model at training sizes 1k–160k using best hyperparameters from Step 2. Writes `outputs/scalability.csv`.

### Step 6 — Plot scalability curves

```bash
python step6_plot_scalability.py
```

Writes `figures/scalability_train_time.png`, `figures/scalability_inference_time.png`, and `figures/scalability_roc_auc.png`.

---

### Quick full run (Linux/macOS)

```bash
bash src/run_overnight.sh      # Steps 0–4 end-to-end
bash src/run_scalability.sh    # Steps 5–6
```

---

## Team

| Member | Role |
| --- | --- |
| Chaz Wilms | Code, report |
| Vaughn Gugger | Code, report, slides |
| Quang Minh Nguyen | Slides |
| Rachel Stevenson | Report |
| Jack Handley | Slides |
| Isaac Avila | Research, report |

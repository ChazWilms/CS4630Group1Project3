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

```
data/
├── HIGGS.csv.gz          # Raw compressed dataset (11M rows, 28 features)
├── X_scaled.csv          # Scaled 200k subsample (from Project 2)
├── y.csv                 # Labels for the subsample (from Project 2)
├── X_pca_10.csv          # PCA-10 features (from Project 2)
└── cluster_labels.csv    # Best-k k-Means cluster IDs (from Project 2)
```

---

## Repository Structure

```
CS4630Group1Project3/
├── data/                        # Raw + Project 2 artifacts
├── src/                         # Pipeline source code
├── splits/                      # Train/test splits
├── models/                      # Trained model files 
├── outputs/                     # Metrics CSVs
├── figures/                     # Plots for the report
├── report/                      # Report drafts and final PDF
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


## Team

| Member | Role |
|---|---|
| Chaz Wilms | TBD |
| Vaughn Gugger | TBD |
| Quang Minh Nguyen | TBD |
| Rachel Stevenson | TBD |
| Jack Handley | TBD |
| Isaac Avila | TBD |

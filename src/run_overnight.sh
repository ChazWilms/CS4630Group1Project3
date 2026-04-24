#!/bin/bash
# Chained overnight run: xgboost (fix 1) -> rbf full 160k (fix 2) -> aggregate -> plots.
# Stops on first error so a broken xgboost run doesn't waste the rbf slot.
set -e
set -o pipefail

cd /Users/chaz/Desktop/CS4630Group1Project3/src
VENV_PY="/Users/chaz/Library/Mobile Documents/com~apple~CloudDocs/School/Current Semester/CS4630/CS4630Group1Project2/venv/bin/python"

echo "=== $(date '+%Y-%m-%d %H:%M:%S'): START xgboost (step2d) ==="
"$VENV_PY" step2d_xgboost_only.py
echo "=== $(date '+%Y-%m-%d %H:%M:%S'): DONE xgboost ==="

echo "=== $(date '+%Y-%m-%d %H:%M:%S'): START rbf full 160k (step2e) ==="
"$VENV_PY" step2e_rbf_full.py
echo "=== $(date '+%Y-%m-%d %H:%M:%S'): DONE rbf full ==="

echo "=== $(date '+%Y-%m-%d %H:%M:%S'): aggregating results (step3) ==="
"$VENV_PY" step3_evaluate.py

echo "=== $(date '+%Y-%m-%d %H:%M:%S'): plotting (step4) ==="
"$VENV_PY" step4_visualize.py

echo "=== $(date '+%Y-%m-%d %H:%M:%S'): ALL DONE ==="

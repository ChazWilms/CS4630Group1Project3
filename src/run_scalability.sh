#!/bin/bash
# Scalability experiment: fixed best params, varying training set size.
set -e
set -o pipefail

cd /Users/chaz/Desktop/CS4630Group1Project3/src
VENV_PY="/Users/chaz/Library/Mobile Documents/com~apple~CloudDocs/School/Current Semester/CS4630/CS4630Group1Project2/venv/bin/python"

echo "=== $(date '+%Y-%m-%d %H:%M:%S'): START step5 scalability ==="
"$VENV_PY" step5_scalability.py
echo "=== $(date '+%Y-%m-%d %H:%M:%S'): DONE step5 ==="

echo "=== $(date '+%Y-%m-%d %H:%M:%S'): START step6 plots ==="
"$VENV_PY" step6_plot_scalability.py
echo "=== $(date '+%Y-%m-%d %H:%M:%S'): DONE step6 ==="

echo "=== $(date '+%Y-%m-%d %H:%M:%S'): ALL DONE ==="

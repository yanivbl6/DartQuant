#!/bin/bash
# Checks if lm-eval datasets are cached; downloads any that are missing.
# Safe to run repeatedly — skips datasets already present.
#
# Usage: ./ensure_datasets.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python "$SCRIPT_DIR/scripts/download_lm_eval_datasets.py" 2>/dev/null | tee /tmp/ds_check.txt

if grep -q "MISSING" /tmp/ds_check.txt; then
    echo ""
    echo "Some datasets are missing. Downloading..."
    echo ""
    HF_DATASETS_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    python "$SCRIPT_DIR/scripts/download_lm_eval_datasets.py"
else
    echo ""
    echo "All datasets are cached."
fi

rm -f /tmp/ds_check.txt

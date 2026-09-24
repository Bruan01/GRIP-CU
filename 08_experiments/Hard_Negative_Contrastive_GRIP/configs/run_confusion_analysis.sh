#!/usr/bin/env bash
# CPU-only description of saved full-vocabulary candidate scores.
# Does not load the 7B teacher and does not train.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
SCORES="${SCORES:-$HNG/results/runs/20260923_offline_confusion_full/candidate_scores.jsonl}"
CONFUSION_DB="${CONFUSION_DB:-$HNG/results/runs/20260923_confusion_db_frozen_b1/confusion_db.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$HNG/results/runs/20260923_offline_confusion_full/analysis}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -e "$RAW_DIR/train.txt" ]]; then
  echo "error: missing raw NELL23K: $RAW_DIR" >&2
  exit 1
fi

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
echo "[launch] confusion analysis -> $OUTPUT_DIR"
"$PYTHON" "$HNG/scripts/analyze_offline_confusion.py" \
  --scores "$SCORES" \
  --confusion_db "$CONFUSION_DB" \
  --raw_dir "$RAW_DIR" \
  --output_dir "$OUTPUT_DIR"
echo "[launch] confusion analysis finished $(date --iso-8601=seconds)"

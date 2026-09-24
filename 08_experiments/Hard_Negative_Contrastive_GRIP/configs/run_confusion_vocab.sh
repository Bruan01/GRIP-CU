#!/usr/bin/env bash
# CPU-only relation-global confusion vocabulary.
# Does not load the 7B teacher, does not train, and does not overwrite
# QA-level analysis/ outputs. MIN_SUPPORT filters figures only.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
SCORES="${SCORES:-$HNG/results/runs/20260923_offline_confusion_full/candidate_scores.jsonl}"
CONFUSION_DB="${CONFUSION_DB:-$HNG/results/runs/20260923_confusion_db_frozen_b1/confusion_db.jsonl}"
DUMP_METADATA="${DUMP_METADATA:-$HNG/results/runs/20260923_offline_confusion_full/metadata.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$HNG/results/runs/20260923_offline_confusion_full/confusion_vocab}"
MIN_SUPPORT="${MIN_SUPPORT:-10}"
NEIGHBOR_K="${NEIGHBOR_K:-20}"
HEATMAP_TOP_N="${HEATMAP_TOP_N:-30}"
NEIGHBORHOOD_PLOT_N="${NEIGHBORHOOD_PLOT_N:-8}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c "import matplotlib" >/dev/null 2>&1; then
  FALLBACK_PYTHON="${FALLBACK_PYTHON:-$(command -v python3 || true)}"
  if [[ -n "$FALLBACK_PYTHON" ]] && "$FALLBACK_PYTHON" -c "import matplotlib, numpy" >/dev/null 2>&1; then
    echo "[launch] $PYTHON lacks matplotlib; using $FALLBACK_PYTHON for figures"
    PYTHON="$FALLBACK_PYTHON"
  else
    echo "[launch] matplotlib unavailable; CSV/JSON will still be written"
  fi
fi
if [[ ! -e "$RAW_DIR/train.txt" ]]; then
  echo "error: missing raw NELL23K: $RAW_DIR" >&2
  exit 1
fi

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
echo "[launch] confusion vocab -> $OUTPUT_DIR min_support=$MIN_SUPPORT"
"$PYTHON" "$HNG/scripts/build_confusion_vocab.py" \
  --scores "$SCORES" \
  --confusion_db "$CONFUSION_DB" \
  --dump_metadata "$DUMP_METADATA" \
  --raw_dir "$RAW_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --min_support "$MIN_SUPPORT" \
  --neighbor_k "$NEIGHBOR_K" \
  --heatmap_top_n "$HEATMAP_TOP_N" \
  --neighborhood_plot_n "$NEIGHBORHOOD_PLOT_N"
echo "[launch] confusion vocab finished $(date --iso-8601=seconds)"

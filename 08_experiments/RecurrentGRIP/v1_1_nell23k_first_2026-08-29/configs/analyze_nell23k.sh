#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  if [[ -f "$VERSION_DIR/results/LAST_NELL23K_PILOT_RUN.txt" ]]; then
    RUN_DIR="$(cat "$VERSION_DIR/results/LAST_NELL23K_PILOT_RUN.txt")"
  elif [[ -f "$VERSION_DIR/results/LAST_NELL23K_SMOKE_RUN.txt" ]]; then
    RUN_DIR="$(cat "$VERSION_DIR/results/LAST_NELL23K_SMOKE_RUN.txt")"
  else
    echo "error: set RUN_DIR to a completed NELL23K run directory" >&2
    exit 1
  fi
fi
if [[ ! -f "$RUN_DIR/predictions.jsonl" ]]; then
  echo "error: predictions not found: $RUN_DIR/predictions.jsonl" >&2
  exit 1
fi

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" scripts/analyze_recurrent_results.py \
  --input_file "$RUN_DIR/predictions.jsonl" \
  --output_dir "$RUN_DIR/analysis" \
  2>&1 | tee "$RUN_DIR/analysis.log"

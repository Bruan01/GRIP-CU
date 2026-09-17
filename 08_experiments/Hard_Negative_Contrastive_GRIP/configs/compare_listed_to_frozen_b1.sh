#!/usr/bin/env bash
# After a listed-only train_graph run, write comparison.json vs frozen B1.
# Does not load a model. Safe to re-run once listed/summary.json exists.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
LISTED_RUN="${LISTED_RUN:-$HNG/results/runs/20260917_qwen7b_train_graph_negatives}"
B1_RUN="${B1_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
INPUT_FILE="${INPUT_FILE:-$HNG/grip_nell23k_tasks.json}"
S1_ADAPTER="${S1_ADAPTER:-$B1_RUN/s1_adapter}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$LISTED_RUN/listed/summary.json" ]]; then
  echo "error: missing listed smoke summary: $LISTED_RUN/listed/summary.json" >&2
  exit 1
fi

LISTED_RUN="$LISTED_RUN" B1_RUN="$B1_RUN" bash "$HNG/configs/attach_frozen_b1.sh"

export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE_DIR"
"$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
  --stage compare \
  --input_file "$INPUT_FILE" \
  --output_dir "$LISTED_RUN" \
  --s1_adapter "$S1_ADAPTER" \
  --model_name qwen-7b \
  --lambda_candidate 1.0
echo "[compare] wrote $LISTED_RUN/comparison.json"

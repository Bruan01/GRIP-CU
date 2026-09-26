#!/usr/bin/env bash
# Freeze Random-K / Top-K Hard / Coverage-Adaptive K from the train-only dump.
# CPU-only. Does not rescore and does not train.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
SCORES="${SCORES:-$HNG/results/runs/20260923_offline_confusion_train_filter/candidate_scores.jsonl}"
METADATA="${METADATA:-$HNG/results/runs/20260923_offline_confusion_train_filter/metadata.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$HNG/results/runs/20260926_shared_pool_samplers}"
K_FIXED="${K_FIXED:-9}"
K_MIN="${K_MIN:-1}"
K_MAX="${K_MAX:-20}"
SEED="${SEED:-2026}"
LOG="${LOG:-$OUTPUT_DIR/run.log}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$SCORES" ]]; then
  echo "error: missing train-only scores: $SCORES" >&2
  exit 1
fi

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
{
  echo "[launch] freeze shared-pool samplers -> $OUTPUT_DIR"
  echo "scores=$SCORES"
  echo "metadata=$METADATA"
  echo "k_fixed=$K_FIXED k_min=$K_MIN k_max=$K_MAX seed=$SEED"
  "$PYTHON" "$HNG/scripts/freeze_shared_pool_samplers.py" \
    --scores "$SCORES" \
    --metadata "$METADATA" \
    --output_dir "$OUTPUT_DIR" \
    --k_fixed "$K_FIXED" \
    --k_min "$K_MIN" \
    --k_max "$K_MAX" \
    --seed "$SEED"
  echo "[launch] freeze finished $(date --iso-8601=seconds)"
} | tee "$LOG"

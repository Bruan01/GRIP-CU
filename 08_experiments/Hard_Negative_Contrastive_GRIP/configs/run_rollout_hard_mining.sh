#!/usr/bin/env bash
# Mine one actual frozen-B1 free-generation error per training QA item.
# This is training-only: no validation/test predictions are read.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
INPUT_FILE="${INPUT_FILE:-$HNG/grip_nell23k_tasks.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
B1_ADAPTER="${B1_ADAPTER:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/b1/adapter}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}_rollout_hard_mining}"
OUTPUT="${OUTPUT:-$RUN_DIR/rollout_hard_manifest.jsonl}"
LIMIT="${LIMIT:-0}"
TMUX_SESSION="${TMUX_SESSION:-rollout-hard-mine-${RUN_ID}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
for required in "$INPUT_FILE" "$RAW_DIR/train.txt" "$RAW_DIR/valid.txt" "$RAW_DIR/test.txt" "$B1_ADAPTER"; do
  if [[ ! -e "$required" ]]; then
    echo "error: missing required path: $required" >&2
    exit 1
  fi
done

source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$RUN_DIR"
{
  echo "input_file=$INPUT_FILE"
  echo "raw_dir=$RAW_DIR"
  echo "b1_adapter=$B1_ADAPTER"
  echo "output=$OUTPUT"
  echo "limit=$LIMIT"
  echo "total_k=${TOTAL_K:-9}"
  echo "max_new_tokens=${MAX_NEW_TOKENS:-32}"
  echo "seed=${SEED:-2026}"
  echo "tmux_session=$TMUX_SESSION"
  date --iso-8601=seconds
} >> "$RUN_DIR/environment.txt"

export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

echo "[launch] rollout-hard mining tmux=$TMUX_SESSION" | tee -a "$RUN_DIR/run.log"
set +e
"$PYTHON" "$HNG/scripts/mine_rollout_hard_negatives.py" \
  --input_file "$INPUT_FILE" \
  --b1_adapter "$B1_ADAPTER" \
  --raw_dir "$RAW_DIR" \
  --output "$OUTPUT" \
  --model_name qwen-7b \
  --model_cache_dir "$CODE_DIR/model_cache" \
  --total_k "${TOTAL_K:-9}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-32}" \
  --seed "${SEED:-2026}" \
  --limit "$LIMIT" \
  --progress_every "${PROGRESS_EVERY:-25}" \
  --resume \
  2>&1 | tee -a "$RUN_DIR/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "[launch] rollout-hard mining finished exit=$rc $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/run.log"
exit "$rc"

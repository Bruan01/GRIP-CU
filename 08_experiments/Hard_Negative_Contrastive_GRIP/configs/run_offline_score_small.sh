#!/usr/bin/env bash
# Small-scale offline full-vocabulary scoring with the frozen B1 adapter.
# Default LIMIT=100 matchable relation QA; set LIMIT up to 1000. Do not use
# this launcher for the full 3253-item dump.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
INPUT_FILE="${INPUT_FILE:-$HNG/grip_nell23k_tasks.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
B1_ADAPTER="${B1_ADAPTER:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/b1/adapter}"
LIMIT="${LIMIT:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}_offline_score_limit${LIMIT}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR}"
TMUX_SESSION="${TMUX_SESSION:-offline-score-${RUN_ID}}"

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

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$OUTPUT_DIR"
{
  echo "task_file=$INPUT_FILE"
  echo "raw_dir=$RAW_DIR"
  echo "b1_adapter=$B1_ADAPTER"
  echo "output_dir=$OUTPUT_DIR"
  echo "limit=$LIMIT"
  echo "temperature=$TEMPERATURE"
  echo "candidate_batch_size=$CANDIDATE_BATCH_SIZE"
  echo "tmux_session=${TMUX_SESSION:-}"
  date --iso-8601=seconds
} >> "$OUTPUT_DIR/environment.txt"

export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

echo "[launch] offline relation scoring limit=$LIMIT tmux=${TMUX_SESSION:-none}" | tee -a "$OUTPUT_DIR/run.log"
set +e
"$PYTHON" "$HNG/scripts/score_relation_candidates_offline.py" \
  --task_file "$INPUT_FILE" \
  --raw_dir "$RAW_DIR" \
  --b1_adapter "$B1_ADAPTER" \
  --output_dir "$OUTPUT_DIR" \
  --model_name qwen-7b \
  --model_cache_dir "$CODE_DIR/model_cache" \
  --limit "$LIMIT" \
  --candidate_batch_size "$CANDIDATE_BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --progress_every "${PROGRESS_EVERY:-5}" \
  ${RESUME:+--resume} \
  2>&1 | tee -a "$OUTPUT_DIR/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "[launch] offline relation scoring finished exit=$rc $(date --iso-8601=seconds)" | tee -a "$OUTPUT_DIR/run.log"
exit "$rc"

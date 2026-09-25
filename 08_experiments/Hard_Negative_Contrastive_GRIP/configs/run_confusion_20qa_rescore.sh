#!/usr/bin/env bash
# Live 20-QA scorer identity check. Loads B1, does not train.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
TASK_FILE="${TASK_FILE:-$HNG/grip_nell23k_tasks.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
B1_ADAPTER="${B1_ADAPTER:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/b1/adapter}"
SCORES="${SCORES:-$HNG/results/runs/20260923_offline_confusion_full/candidate_scores.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$HNG/results/runs/20260923_offline_confusion_full/audit}"
OUTPUT="${OUTPUT:-$OUTPUT_DIR/live_20qa_rescore.json}"
QA_IDS="${QA_IDS:-$OUTPUT_DIR/sampled_scorer_qa_ids.json}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-8}"
TMUX_SESSION="${TMUX_SESSION:-confusion-audit-20qa-$(date +%Y%m%d)}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
for required in "$TASK_FILE" "$RAW_DIR/train.txt" "$B1_ADAPTER" "$SCORES"; do
  if [[ ! -e "$required" ]]; then
    echo "error: missing required path: $required" >&2
    exit 1
  fi
done

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

extra=()
if [[ -f "$QA_IDS" ]]; then
  extra+=(--qa_ids "$QA_IDS")
fi

echo "[launch] 20-QA live rescore -> $OUTPUT tmux=${TMUX_SESSION:-none}" | tee -a "$OUTPUT_DIR/live_20qa_rescore.log"
set +e
"$PYTHON" "$HNG/scripts/rescore_confusion_20qa.py" \
  --scores "$SCORES" \
  --task_file "$TASK_FILE" \
  --raw_dir "$RAW_DIR" \
  --b1_adapter "$B1_ADAPTER" \
  --output "$OUTPUT" \
  --model_name qwen-7b \
  --model_cache_dir "$CODE_DIR/model_cache" \
  --candidate_batch_size "$CANDIDATE_BATCH_SIZE" \
  "${extra[@]}" \
  2>&1 | tee -a "$OUTPUT_DIR/live_20qa_rescore.log"
rc=${PIPESTATUS[0]}
set -e
echo "[launch] 20-QA live rescore finished exit=$rc $(date --iso-8601=seconds)" | tee -a "$OUTPUT_DIR/live_20qa_rescore.log"
exit "$rc"

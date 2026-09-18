#!/usr/bin/env bash
# Full official NELL23K val/test decode on saved 7B B1/listed adapters.
#
# Does not train. Writes generate + closed-set summaries under a new run dir.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
ADAPTER_RUN="${ADAPTER_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
FULL_EVAL="${FULL_EVAL:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FULL_DIR="${FULL_DIR:-$HNG/results/runs/${RUN_ID}_qwen7b_full_decode}"
PROGRESS_EVERY="${PROGRESS_EVERY:-64}"
TMUX_SESSION="${TMUX_SESSION:-full-decode-${RUN_ID}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$ADAPTER_RUN/b1/adapter" || ! -d "$ADAPTER_RUN/listed/adapter" ]]; then
  echo "error: missing 7B adapters under $ADAPTER_RUN" >&2
  exit 1
fi

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

if [[ ! -f "$FULL_EVAL" ]]; then
  echo "[full-eval] missing $FULL_EVAL; preparing"
  bash "$HNG/configs/prepare_full_nell23k_eval.sh"
fi
if [[ ! -f "$FULL_EVAL" ]]; then
  echo "error: full eval file not found: $FULL_EVAL" >&2
  exit 1
fi

mkdir -p "$FULL_DIR/b1" "$FULL_DIR/listed"
{
  echo "adapter_run=$ADAPTER_RUN"
  echo "full_dir=$FULL_DIR"
  echo "eval_file=$FULL_EVAL"
  echo "progress_every=$PROGRESS_EVERY"
  date --iso-8601=seconds
} | tee -a "$FULL_DIR/environment.txt"

export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

eval_one() {
  local adapter_dir="$1"
  local output_dir="$2"
  echo "[decode] adapter=$adapter_dir decode=both eval=$FULL_EVAL"
  "$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
    --adapter_dir "$adapter_dir" \
    --output_dir "$output_dir" \
    --eval_file "$FULL_EVAL" \
    --model_name qwen-7b \
    --model_cache_dir "$CODE_DIR/model_cache" \
    --decode both \
    --gen_max_length 32 \
    --progress_every "$PROGRESS_EVERY" \
    --eval_name full
}

echo "[decode] full generate+closed-set start $(date --iso-8601=seconds)" | tee -a "$FULL_DIR/decode.log"
eval_one "$ADAPTER_RUN/b1/adapter" "$FULL_DIR/b1" 2>&1 | tee -a "$FULL_DIR/decode.log"
eval_one "$ADAPTER_RUN/listed/adapter" "$FULL_DIR/listed" 2>&1 | tee -a "$FULL_DIR/decode.log"
"$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
  --compare \
  --eval_file "$FULL_EVAL" \
  --output_dir "$FULL_DIR" \
  --eval_name full 2>&1 | tee -a "$FULL_DIR/decode.log"
echo "$FULL_DIR" > "$HNG/results/LAST_FULL_DECODE_RUN.txt"
echo "[decode] full done $(date --iso-8601=seconds)" | tee -a "$FULL_DIR/decode.log"
echo "full decode completed adapter=$ADAPTER_RUN full=$FULL_DIR"

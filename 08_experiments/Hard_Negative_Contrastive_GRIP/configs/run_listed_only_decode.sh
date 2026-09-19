#!/usr/bin/env bash
# Decode the experiment-H listed adapter only. Reuse frozen B1 predictions
# from the 20260915 decode (B1 is generation-only and was not retrained).
#
# SCALE=pilot|full. DECODE=generate|closed_set|both.
# WAIT_FOR_PATTERN waits until no python process matches that cmdline
# substring, so this can queue behind an occupying GPU job.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
SCALE="${SCALE:-pilot}"
DECODE="${DECODE:-both}"
ADAPTER_RUN="${ADAPTER_RUN:-$HNG/results/runs/20260918_qwen7b_train_graph_negatives}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROGRESS_EVERY="${PROGRESS_EVERY:-64}"
LAST_RUN_FILE="$HNG/results/LAST_LISTED_ONLY_DECODE_RUN.txt"

case "$SCALE" in
  pilot)
    EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_pilot.aligned.json}"
    B1_DECODE_RUN="${B1_DECODE_RUN:-$HNG/results/runs/20260915_101030_qwen7b_pilot_decode}"
    DEFAULT_DIR="$HNG/results/runs/${RUN_ID}_qwen7b_pilot_listed_only"
    ;;
  full)
    EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
    B1_DECODE_RUN="${B1_DECODE_RUN:-$HNG/results/runs/20260915_140500_qwen7b_full_decode}"
    DEFAULT_DIR="$HNG/results/runs/${RUN_ID}_qwen7b_full_listed_only"
    ;;
  *)
    echo "error: SCALE must be pilot or full, got $SCALE" >&2
    exit 1
    ;;
esac

RUN_DIR="${RUN_DIR:-$DEFAULT_DIR}"
if [[ "${RESUME_LAST:-0}" == "1" ]]; then
  if [[ ! -f "$LAST_RUN_FILE" ]]; then
    echo "error: RESUME_LAST=1 but missing $LAST_RUN_FILE" >&2
    exit 1
  fi
  RUN_DIR="$(tr -d '\n' < "$LAST_RUN_FILE")"
fi
TMUX_SESSION="${TMUX_SESSION:-listed-only-${SCALE}-${RUN_ID}}"
LISTED_ADAPTER="${LISTED_ADAPTER:-$ADAPTER_RUN/listed/adapter}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$LISTED_ADAPTER" ]]; then
  echo "error: missing listed adapter: $LISTED_ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$B1_DECODE_RUN/b1/summary.json" ]]; then
  echo "error: missing frozen B1 generate summary: $B1_DECODE_RUN/b1/summary.json" >&2
  exit 1
fi
if [[ "$DECODE" == "both" || "$DECODE" == "closed_set" ]]; then
  if [[ ! -f "$B1_DECODE_RUN/b1/summary_closed_set.json" ]]; then
    echo "error: missing frozen B1 closed-set summary: $B1_DECODE_RUN/b1/summary_closed_set.json" >&2
    exit 1
  fi
fi
if [[ ! -f "$EVAL_FILE" ]]; then
  echo "error: eval file not found: $EVAL_FILE" >&2
  exit 1
fi

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

wait_for_gpu_holder() {
  local pattern="${WAIT_FOR_PATTERN:-}"
  if [[ -z "$pattern" ]]; then
    return 0
  fi
  echo "[wait] until no python cmdline contains: $pattern" | tee -a "$RUN_DIR/decode.log"
  while pgrep -af python | grep -F "$pattern" | grep -v grep >/dev/null; do
    echo "[wait] still occupied $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/decode.log"
    sleep 60
  done
  echo "[wait] clear $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/decode.log"
}

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$LAST_RUN_FILE"
{
  echo "scale=$SCALE"
  echo "decode=$DECODE"
  echo "adapter_run=$ADAPTER_RUN"
  echo "listed_adapter=$LISTED_ADAPTER"
  echo "b1_decode_run=$B1_DECODE_RUN"
  echo "eval_file=$EVAL_FILE"
  echo "run_dir=$RUN_DIR"
  echo "wait_for_pattern=${WAIT_FOR_PATTERN:-}"
  echo "tmux_session=${TMUX_SESSION:-}"
  date --iso-8601=seconds
} | tee -a "$RUN_DIR/environment.txt"

wait_for_gpu_holder

LISTED_RUN="$RUN_DIR" B1_RUN="$B1_DECODE_RUN" bash "$HNG/configs/attach_frozen_b1.sh"

export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

RESUME_FLAG=()
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  RESUME_FLAG+=(--no_resume)
fi

echo "[decode] listed-only start scale=$SCALE decode=$DECODE $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/decode.log"
"$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
  --adapter_dir "$LISTED_ADAPTER" \
  --output_dir "$RUN_DIR/listed" \
  --eval_file "$EVAL_FILE" \
  --model_name qwen-7b \
  --model_cache_dir "$CODE_DIR/model_cache" \
  --decode "$DECODE" \
  --gen_max_length 32 \
  --progress_every "$PROGRESS_EVERY" \
  --eval_name "$SCALE" \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee -a "$RUN_DIR/decode.log"

"$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
  --compare \
  --eval_file "$EVAL_FILE" \
  --output_dir "$RUN_DIR" \
  --eval_name "$SCALE" \
  2>&1 | tee -a "$RUN_DIR/decode.log"
echo "[decode] listed-only done $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/decode.log"
echo "listed-only decode completed listed=$LISTED_ADAPTER b1=$B1_DECODE_RUN run=$RUN_DIR"

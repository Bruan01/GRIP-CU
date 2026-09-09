#!/usr/bin/env bash
# Original GRIP vs GRIP + listed 10-way contrastive.
#
# Shared Stage 1 stores the NELL23K graph in MLP-LoRA (same recipe as
# quick01_storage_quick). Stage 2 then forks:
#   b1     = generation loss only
#   listed = generation + InfoNCE over the prompt's 9 distractors
#
# SCALE=smoke|pilot. One GPU runs the forks sequentially; two GPUs run them
# in parallel after Stage 1. This is the method gate, not a paper result.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
SCALE="${SCALE:-smoke}"
PARALLEL_S2="${PARALLEL_S2:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}_listed_vs_b1_${SCALE}}"

case "$SCALE" in
  smoke)
    INPUT_FILE="${INPUT_FILE:-$HNG/data/nell23k/recurrent_relation_prediction.aligned.json}"
    ;;
  pilot)
    INPUT_FILE="${INPUT_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_pilot.aligned.json}"
    ;;
  *)
    echo "error: SCALE must be smoke or pilot, got: $SCALE" >&2
    exit 1
    ;;
esac

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "error: RUN_ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "error: aligned input not found: $INPUT_FILE" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "error: run directory already exists; choose a new RUN_ID: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "scale=$SCALE"
  echo "input_file=$INPUT_FILE"
  date --iso-8601=seconds
  nvidia-smi || true
} > "$RUN_DIR/environment.txt" 2>&1

export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

COMMON=(
  "$PYTHON" "$HNG/scripts/train_listed_contrastive.py"
  --input_file "$INPUT_FILE"
  --output_dir "$RUN_DIR"
  --model_name qwen-0.5b
  --model_cache_dir "$CODE_DIR/model_cache"
  --lora_r 4
  --lora_alpha 8
  --target_modules down_proj up_proj gate_proj
  --num_train_epochs 1
  --involve_qa_epochs 10
  --s1_stop_loss_threshold 0.15
  --per_device_train_batch_size 8
  --gradient_accumulation_steps 64
  --learning_rate 1e-3
  --weight_decay 1e-4
  --max_grad_norm 1.0
  --lambda_candidate 1.0
  --temperature 1.0
  --gen_max_length 32
  --seed 2026
)

NGPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ -z "$NGPU" ]]; then
  NGPU=0
fi

run_stage() {
  local stage="$1"
  shift
  echo "[launch] stage=$stage $*" | tee -a "$RUN_DIR/run.log"
  "${COMMON[@]}" --stage "$stage" "$@" 2>&1 | tee -a "$RUN_DIR/run.log"
}

if [[ "$NGPU" -ge 2 && "$PARALLEL_S2" == "1" ]]; then
  run_stage s1
  echo "[launch] forking Stage 2 onto GPU 0 (b1) and GPU 1 (listed)" | tee -a "$RUN_DIR/run.log"
  CUDA_VISIBLE_DEVICES=0 "${COMMON[@]}" --stage b1 --s1_adapter "$RUN_DIR/s1_adapter" \
    > "$RUN_DIR/b1_train.log" 2>&1 &
  pid_b1=$!
  CUDA_VISIBLE_DEVICES=1 "${COMMON[@]}" --stage listed --s1_adapter "$RUN_DIR/s1_adapter" \
    > "$RUN_DIR/listed_train.log" 2>&1 &
  pid_listed=$!
  status=0
  wait "$pid_b1" || status=1
  wait "$pid_listed" || status=1
  cat "$RUN_DIR/b1_train.log" "$RUN_DIR/listed_train.log" >> "$RUN_DIR/run.log"
  if [[ "$status" -ne 0 ]]; then
    echo "error: a Stage-2 fork failed" | tee -a "$RUN_DIR/run.log" >&2
    exit 1
  fi
  run_stage compare --s1_adapter "$RUN_DIR/s1_adapter"
else
  run_stage all
fi

echo "$RUN_DIR" > "$HNG/results/LAST_LISTED_VS_B1_RUN.txt"
echo "listed vs original GRIP completed: $RUN_DIR"

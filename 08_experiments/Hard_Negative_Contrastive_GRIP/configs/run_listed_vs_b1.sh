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
MODEL_NAME="${MODEL_NAME:-qwen-0.5b}"
PARALLEL_S2="${PARALLEL_S2:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
RESUME_S1_ADAPTER="${RESUME_S1_ADAPTER:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODEL_NAME//./}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}_listed_vs_b1_${MODEL_TAG}_${SCALE}}"

case "$MODEL_NAME" in
  qwen-7b)
    # Paper 7B recipe on one 24GB 3090: batch 1, effective 512.
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-512}"
    S1_CHECKPOINT_FLAG=(--s1_gradient_checkpointing)
    ;;
  *)
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
    S1_CHECKPOINT_FLAG=()
    ;;
esac

case "$SCALE" in
  smoke)
    ALIGNED_EVAL="${ALIGNED_EVAL:-$HNG/data/nell23k/recurrent_relation_prediction.aligned.json}"
    ;;
  pilot)
    ALIGNED_EVAL="${ALIGNED_EVAL:-$HNG/data/nell23k/recurrent_relation_prediction_pilot.aligned.json}"
    ;;
  *)
    echo "error: SCALE must be smoke or pilot, got: $SCALE" >&2
    exit 1
    ;;
esac
PAPER_TASKS="$HNG/grip_nell23k_tasks.json"
if [[ "$MODEL_NAME" == "qwen-7b" ]]; then
  # Paper 7B generated context+QA. Val/test EM stays on the aligned 10-way split.
  INPUT_FILE="${INPUT_FILE:-$PAPER_TASKS}"
  EVAL_FILE="${EVAL_FILE:-$ALIGNED_EVAL}"
else
  INPUT_FILE="${INPUT_FILE:-$ALIGNED_EVAL}"
  EVAL_FILE="${EVAL_FILE:-$INPUT_FILE}"
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "error: RUN_ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "error: training input not found: $INPUT_FILE" >&2
  exit 1
fi
if [[ ! -f "$EVAL_FILE" ]]; then
  echo "error: eval file not found: $EVAL_FILE" >&2
  exit 1
fi
MODEL_CACHE="$CODE_DIR/model_cache"
if [[ "$MODEL_NAME" == "qwen-7b" ]]; then
  MODEL_DIR="$MODEL_CACHE/Qwen--Qwen2.5-7B-Instruct"
  if ! PYTHONPATH="$CODE_DIR" "$PYTHON" -c "
from pathlib import Path
from models.utils import _is_complete_model_dir
raise SystemExit(0 if _is_complete_model_dir(Path(r'$MODEL_DIR')) else 1)
"; then
    echo "error: Qwen2.5-7B cache missing or incomplete: $MODEL_DIR" >&2
    echo "Download first:" >&2
    echo "  cd $CODE_DIR && PYTHONPATH=. $PYTHON scripts/download_model.py --model_name qwen-7b --model_cache_dir model_cache" >&2
    exit 1
  fi
fi
if [[ -n "$RESUME_S1_ADAPTER" && ! -d "$RESUME_S1_ADAPTER" ]]; then
  echo "error: missing Stage-1 adapter: $RESUME_S1_ADAPTER" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" && -z "$RESUME_S1_ADAPTER" ]]; then
  echo "error: run directory already exists; choose a new RUN_ID: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "scale=$SCALE"
  echo "model_name=$MODEL_NAME"
  echo "per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
  echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  echo "input_file=$INPUT_FILE"
  echo "eval_file=$EVAL_FILE"
  echo "resume_s1_adapter=${RESUME_S1_ADAPTER:-}"
  echo "skip_train=${SKIP_TRAIN:-0}"
  date --iso-8601=seconds
  nvidia-smi || true
} >> "$RUN_DIR/environment.txt" 2>&1

export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

COMMON=(
  "$PYTHON" "$HNG/scripts/train_listed_contrastive.py"
  --input_file "$INPUT_FILE"
  --eval_file "$EVAL_FILE"
  --output_dir "$RUN_DIR"
  --model_name "$MODEL_NAME"
  --model_cache_dir "$MODEL_CACHE"
  --lora_r 4
  --lora_alpha 8
  --target_modules down_proj up_proj gate_proj
  --num_train_epochs 1
  --involve_qa_epochs 10
  --s1_stop_loss_threshold 0.15
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  "${S1_CHECKPOINT_FLAG[@]}"
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
  NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
fi
if [[ -z "$NGPU" || "$NGPU" == "0" || ! "$NGPU" =~ ^[0-9]+$ ]]; then
  NGPU="$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)"
fi
if [[ -z "$NGPU" || ! "$NGPU" =~ ^[0-9]+$ ]]; then
  NGPU=0
fi

run_stage() {
  local stage="$1"
  shift
  echo "[launch] stage=$stage $*" | tee -a "$RUN_DIR/run.log"
  set +e
  "${COMMON[@]}" --stage "$stage" "$@" 2>&1 | tee -a "$RUN_DIR/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[launch] Stage $stage finished exit=$rc $(date --iso-8601=seconds)" | tee -a "$RUN_DIR/run.log"
  return "$rc"
}

if [[ -n "$RESUME_S1_ADAPTER" ]]; then
  echo "[launch] resume Stage 2 sequentially from $RESUME_S1_ADAPTER ngpu=$NGPU skip_train=$SKIP_TRAIN" | tee -a "$RUN_DIR/run.log"
  b1_args=(--s1_adapter "$RESUME_S1_ADAPTER")
  listed_args=(--s1_adapter "$RESUME_S1_ADAPTER")
  if [[ "$SKIP_TRAIN" == "1" && -d "$RUN_DIR/b1/adapter" ]]; then
    b1_args+=(--skip_train)
  fi
  if [[ "$SKIP_TRAIN" == "1" && -d "$RUN_DIR/listed/adapter" ]]; then
    listed_args+=(--skip_train)
  fi
  run_stage b1 "${b1_args[@]}"
  run_stage listed "${listed_args[@]}"
  run_stage compare --s1_adapter "$RESUME_S1_ADAPTER"
elif [[ "$NGPU" -ge 2 && "$PARALLEL_S2" == "1" ]]; then
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

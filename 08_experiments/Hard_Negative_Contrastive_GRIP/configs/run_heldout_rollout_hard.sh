#!/usr/bin/env bash
# Held-out rollout mining experiment:
# A trains the frozen B1 teacher; B is mining-only; listed trains on A with
# one transferred rollout error plus eight uniform train-graph negatives.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
SCALE="${SCALE:-smoke}"
TASK_FILE="${TASK_FILE:-$HNG/grip_nell23k_tasks.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
OLD_RUN="${OLD_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
S1_ADAPTER="${S1_ADAPTER:-$OLD_RUN/s1_adapter}"
MODEL_NAME="${MODEL_NAME:-qwen-7b}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_heldout_rollout_${SCALE}}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}}"
TMUX_SESSION="${TMUX_SESSION:-heldout-rollout-${SCALE}-$(date +%Y%m%d)}"
SEED="${SEED:-2026}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.2}"

case "$SCALE" in
  smoke)
    EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction.aligned.json}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-96}"
    MAX_HOLDOUT_SAMPLES="${MAX_HOLDOUT_SAMPLES:-64}"
    ;;
  pilot)
    EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_pilot.aligned.json}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
    MAX_HOLDOUT_SAMPLES="${MAX_HOLDOUT_SAMPLES:-512}"
    ;;
  full)
    EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
    MAX_HOLDOUT_SAMPLES="${MAX_HOLDOUT_SAMPLES:-0}"
    ;;
  *)
    echo "error: SCALE must be smoke, pilot, or full, got: $SCALE" >&2
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
for required in "$TASK_FILE" "$RAW_DIR/train.txt" "$RAW_DIR/valid.txt" "$RAW_DIR/test.txt" "$S1_ADAPTER" "$EVAL_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "error: missing required path: $required" >&2
    exit 1
  fi
done
if [[ "${FORCE_NEW:-0}" == "1" && -e "$RUN_DIR" ]]; then
  echo "error: FORCE_NEW=1 but run directory already exists: $RUN_DIR" >&2
  exit 1
fi

source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$RUN_DIR"
TRAIN_TASK="$RUN_DIR/task_train.json"
HOLDOUT_TASK="$RUN_DIR/task_holdout.json"
SOURCE_MANIFEST="$RUN_DIR/heldout_rollout_manifest.jsonl"
TARGET_MANIFEST="$RUN_DIR/target_rollout_manifest.jsonl"
TEACHER_DIR="$RUN_DIR/teacher_b1"
LISTED_DIR="$RUN_DIR/listed"

{
  echo "scale=$SCALE"
  echo "model_name=$MODEL_NAME"
  echo "task_file=$TASK_FILE"
  echo "train_task=$TRAIN_TASK"
  echo "holdout_task=$HOLDOUT_TASK"
  echo "source_manifest=$SOURCE_MANIFEST"
  echo "target_manifest=$TARGET_MANIFEST"
  echo "s1_adapter=$S1_ADAPTER"
  echo "eval_file=$EVAL_FILE"
  echo "holdout_fraction=$HOLDOUT_FRACTION"
  echo "max_train_samples=$MAX_TRAIN_SAMPLES"
  echo "max_holdout_samples=$MAX_HOLDOUT_SAMPLES"
  echo "seed=$SEED"
  echo "tmux_session=$TMUX_SESSION"
  date --iso-8601=seconds
} >> "$RUN_DIR/environment.txt"

export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

log() {
  echo "$*" | tee -a "$RUN_DIR/run.log"
}

if [[ ! -f "$TRAIN_TASK" || ! -f "$HOLDOUT_TASK" ]]; then
  log "[heldout] split task QA"
  "$PYTHON" "$HNG/scripts/split_task_qa.py" \
    --input "$TASK_FILE" \
    --train-output "$TRAIN_TASK" \
    --holdout-output "$HOLDOUT_TASK" \
    --holdout-fraction "$HOLDOUT_FRACTION" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    --max-holdout-samples "$MAX_HOLDOUT_SAMPLES" \
    --seed "$SEED" 2>&1 | tee -a "$RUN_DIR/run.log"
fi

COMMON=(
  "$PYTHON" "$HNG/scripts/train_listed_contrastive.py"
  --input_file "$TRAIN_TASK"
  --eval_file "$EVAL_FILE"
  --model_name "$MODEL_NAME"
  --model_cache_dir "$CODE_DIR/model_cache"
  --lora_r 4
  --lora_alpha 8
  --target_modules down_proj up_proj gate_proj
  --num_train_epochs 1
  --involve_qa_epochs 10
  --s1_stop_loss_threshold 0.15
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 512
  --s1_gradient_checkpointing
  --learning_rate 1e-3
  --weight_decay 1e-4
  --max_grad_norm 1.0
  --lambda_candidate 0.0
  --temperature 1.0
  --gen_max_length 32
  --seed "$SEED"
  --raw_dir "$RAW_DIR"
  --save_steps "${SAVE_STEPS:-10}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
)

if [[ ! -f "$TEACHER_DIR/b1/adapter/adapter_config.json" ]]; then
  log "[heldout] train frozen B1 teacher on disjoint train QA"
  set +e
  "${COMMON[@]}" \
    --output_dir "$TEACHER_DIR" \
    --stage b1 \
    --s1_adapter "$S1_ADAPTER" \
    2>&1 | tee -a "$RUN_DIR/run.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "[heldout] teacher failed exit=$rc"
    exit "$rc"
  fi
else
  log "[heldout] reuse teacher adapter $TEACHER_DIR/b1/adapter"
fi

if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  log "[heldout] mine held-out rollout errors"
  "$PYTHON" "$HNG/scripts/mine_rollout_hard_negatives.py" \
    --input_file "$HOLDOUT_TASK" \
    --b1_adapter "$TEACHER_DIR/b1/adapter" \
    --raw_dir "$RAW_DIR" \
    --output "$SOURCE_MANIFEST" \
    --model_name "$MODEL_NAME" \
    --model_cache_dir "$CODE_DIR/model_cache" \
    --total_k 9 \
    --max_new_tokens 32 \
    --seed "$SEED" \
    --progress_every "${PROGRESS_EVERY:-25}" \
    --resume 2>&1 | tee -a "$RUN_DIR/run.log"
fi

if [[ ! -f "$TARGET_MANIFEST" ]]; then
  log "[heldout] transfer global hard pool onto train QA"
  "$PYTHON" "$HNG/scripts/build_heldout_rollout_manifest.py" \
    --source-manifest "$SOURCE_MANIFEST" \
    --target-task "$TRAIN_TASK" \
    --raw-dir "$RAW_DIR" \
    --output "$TARGET_MANIFEST" \
    --total-k 9 \
    --hard-k 1 \
    --seed "$SEED" 2>&1 | tee -a "$RUN_DIR/run.log"
fi

if [[ ! -f "$LISTED_DIR/adapter/adapter_config.json" ]]; then
  log "[heldout] train listed on train QA with 1 hard + 8 uniform"
  set +e
  "$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
    --input_file "$TRAIN_TASK" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$RUN_DIR" \
    --stage listed \
    --s1_adapter "$S1_ADAPTER" \
    --model_name "$MODEL_NAME" \
    --model_cache_dir "$CODE_DIR/model_cache" \
    --lora_r 4 \
    --lora_alpha 8 \
    --target_modules down_proj up_proj gate_proj \
    --num_train_epochs 1 \
    --involve_qa_epochs 10 \
    --s1_stop_loss_threshold 0.15 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 512 \
    --s1_gradient_checkpointing \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --lambda_candidate 1.0 \
    --temperature 1.0 \
    --gen_max_length 32 \
    --seed "$SEED" \
    --listed_negative_k 9 \
    --listed_negative_source rollout_hard \
    --score_hard_manifest "$TARGET_MANIFEST" \
    --raw_dir "$RAW_DIR" \
    --save_steps "${SAVE_STEPS:-10}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    2>&1 | tee -a "$RUN_DIR/run.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "[heldout] listed training failed exit=$rc"
    exit "$rc"
  fi
else
  log "[heldout] reuse listed adapter $LISTED_DIR/adapter"
fi

log "[heldout] smoke/eval decode with candidate-valid monitoring"
DECODE_DIR="$RUN_DIR/decode"
mkdir -p "$DECODE_DIR/b1" "$DECODE_DIR/listed"
for variant in b1 listed; do
  adapter="$TEACHER_DIR/b1/adapter"
  [[ "$variant" == "listed" ]] && adapter="$LISTED_DIR/adapter"
  "$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
    --adapter_dir "$adapter" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$DECODE_DIR/$variant" \
    --model_name "$MODEL_NAME" \
    --model_cache_dir "$CODE_DIR/model_cache" \
    --decode both \
    --gen_max_length 32 \
    --progress_every "${DECODE_PROGRESS_EVERY:-32}" 2>&1 | tee -a "$RUN_DIR/run.log"
done
"$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
  --compare \
  --eval_file "$EVAL_FILE" \
  --output_dir "$DECODE_DIR" \
  --eval_name "$SCALE" 2>&1 | tee -a "$RUN_DIR/run.log"

echo "$RUN_DIR" > "$HNG/results/LAST_HELDOUT_ROLLOUT_RUN.txt"
log "[heldout] completed run_dir=$RUN_DIR"

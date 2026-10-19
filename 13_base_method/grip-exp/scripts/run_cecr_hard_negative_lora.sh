#!/usr/bin/env bash
# CECR E3: train a GRIP LoRA adapter with format-aligned hard-negative QA.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-model_cache}"
TASK_FILE="${TASK_FILE:-artifacts/nell23k_cecr_hard_negative/tasks.json}"
BASE_TASK_FILE="${BASE_TASK_FILE:-artifacts/nell23k_bridge_explicit_7b/tasks.json}"
ADAPTER_DIR="${ADAPTER_DIR:-artifacts/nell23k_cecr_hard_negative/lora_adapter}"
TRAINER_OUTPUT_DIR="${TRAINER_OUTPUT_DIR:-artifacts/nell23k_cecr_hard_negative/trainer}"
NUM_QA="${NUM_QA:-12014}"
RELATION_CARD_MODE="${RELATION_CARD_MODE:-none}"
STAGE2_MIX_CONTEXT="${STAGE2_MIX_CONTEXT:-true}"
STAGE2_CONTEXT_REPLAY_RATIO="${STAGE2_CONTEXT_REPLAY_RATIO:-1.0}"

STAGE2_MIX_ARGS=()
case "${STAGE2_MIX_CONTEXT,,}" in
  1|true|yes|y|on) STAGE2_MIX_ARGS+=(--stage2-mix-context) ;;
  0|false|no|n|off) STAGE2_MIX_ARGS+=(--no-stage2-mix-context) ;;
  *) echo "Invalid STAGE2_MIX_CONTEXT=$STAGE2_MIX_CONTEXT (expected true/false)" >&2; exit 2 ;;
esac

if [[ ! -f "$TASK_FILE" ]]; then
  "$PYTHON_EXECUTABLE" scripts/build_cecr_hard_negative_qa.py \
    --train-triples-file outputs/data/nell23k/train.txt \
    --base-task-file "$BASE_TASK_FILE" \
    --out-task-file "$TASK_FILE" \
    --num-qa "$NUM_QA" \
    --seed 2026 \
    --relation-card-mode "$RELATION_CARD_MODE" \
    --preview 0
fi

exec "$PYTHON_EXECUTABLE" scripts/train_nell23k_lora.py \
  --task-file "$TASK_FILE" \
  --adapter-dir "$ADAPTER_DIR" \
  --trainer-output-dir "$TRAINER_OUTPUT_DIR" \
  --overwrite-adapter \
  --seed 2026 \
  --model-name qwen-7b \
  --model-source local \
  --model-cache-dir "$MODEL_CACHE_DIR" \
  --local-files-only \
  --tokenize-max-length 4096 \
  --lora-r 4 \
  --lora-alpha 8 \
  --target-modules down_proj up_proj gate_proj \
  --num-train-epochs 1 \
  --involve-qa-epochs 10 \
  "${STAGE2_MIX_ARGS[@]}" \
  --stage2-context-replay-ratio "$STAGE2_CONTEXT_REPLAY_RATIO" \
  --s1-stop-loss-threshold 0.15 \
  --s2-stop-loss-threshold 0.15 \
  --s1-min-epoch 1 \
  --s2-min-epoch 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 512 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --adam-epsilon 1e-8 \
  --max-grad-norm 1.0

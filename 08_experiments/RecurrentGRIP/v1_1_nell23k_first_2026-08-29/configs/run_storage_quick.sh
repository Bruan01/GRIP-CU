#!/usr/bin/env bash
# Storage-only 0.5B quick validation: does MLP-LoRA (full layers) + two-stage
# training make "correct adapter" beat "no adapter" on NELL23K relation prediction?
#
# This is the cheap check from REPRODUCTION_RECIPE.md before committing to a 7B run.
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
INPUT_FILE="${INPUT_FILE:-$CODE_DIR/outputs/data/nell23k/recurrent_relation_prediction_pilot.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$VERSION_DIR/results/runs/${RUN_ID}_storage_quick}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "error: input file not found: $INPUT_FILE" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$CODE_DIR"

"$PYTHON" scripts/quick_validate_storage.py \
  --input_file "$INPUT_FILE" \
  --output_dir "$RUN_DIR" \
  --model_name qwen-0.5b \
  --model_cache_dir model_cache \
  --lora_r 4 \
  --lora_alpha 8 \
  --target_modules down_proj up_proj gate_proj \
  --num_train_epochs 1 \
  --involve_qa_epochs 10 \
  --s1_stop_loss_threshold 0.15 \
  --s2_stop_loss_threshold 0.15 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 64 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --max_grad_norm 1.0 \
  --gen_max_length 32 \
  --seed 2026 \
  2>&1 | tee "$RUN_DIR/run.log"

echo "$RUN_DIR" > "$VERSION_DIR/results/LAST_STORAGE_QUICK_RUN.txt"
echo "Storage quick validation completed: $RUN_DIR"

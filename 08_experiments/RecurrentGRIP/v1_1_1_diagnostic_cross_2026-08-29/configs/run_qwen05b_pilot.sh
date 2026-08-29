#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
CLEGR_INPUT="${CLEGR_INPUT:-$CODE_DIR/outputs/data/clegr_reasoning/recurrent_station_shortest.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$VERSION_DIR/results/runs/${RUN_ID}_qwen05b_pilot}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "error: RUN_ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "error: run configs/setup_wsl3090.sh first" >&2
  exit 1
fi
if [[ ! -f "$CLEGR_INPUT" ]]; then
  echo "error: CLEGR input not found: $CLEGR_INPUT" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "error: run directory already exists; choose a new RUN_ID: $RUN_DIR" >&2
  exit 1
fi
command -v timeout >/dev/null 2>&1 || { echo "error: GNU timeout is required in WSL2" >&2; exit 1; }

mkdir -p "$RUN_DIR"
cp "$VERSION_DIR/configs/pilot_qwen05b.json" "$RUN_DIR/config.json"
{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "clegr_input=$CLEGR_INPUT"
  date --iso-8601=seconds
  nvidia-smi
  "$PYTHON" "$VERSION_DIR/configs/verify_wsl3090.py"
  "$PYTHON" -m pip freeze 2>/dev/null || uv pip freeze --python "$PYTHON"
} > "$RUN_DIR/environment.txt" 2>&1

cd "$CODE_DIR"
timeout --signal=TERM --kill-after=5m 180m \
  "$PYTHON" scripts/run_recurrent_pilot.py \
    --input_file "$CLEGR_INPUT" \
    --output_file "$RUN_DIR/predictions.jsonl" \
    --training_output_dir "$RUN_DIR/trainer" \
    --model_name qwen-0.5b \
    --max_graphs 16 \
    --involve_qa_epochs 1 \
    --gen_max_length 32 \
    --wall_time_limit_minutes 120 \
    --hard_stop_minutes 180 \
    --seed 2026 \
    -- \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --bf16 true \
    --recurrent_depth_train 2 \
    --recurrent_depth_sweep 1 2 3 4 5 \
    --target_modules q_proj k_proj v_proj \
    --adapter_control all \
    --evaluation_device cuda \
    --require_cuda true \
    --adapter_output_dir "$RUN_DIR/adapters" \
  2>&1 | tee "$RUN_DIR/run.log"

echo "$RUN_DIR" > "$VERSION_DIR/results/LAST_PILOT_RUN.txt"
echo "pilot completed: $RUN_DIR"

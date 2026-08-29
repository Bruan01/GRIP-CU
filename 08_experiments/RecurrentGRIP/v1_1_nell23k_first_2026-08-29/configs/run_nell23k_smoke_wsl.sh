#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
NELL_INPUT="${NELL_INPUT:-$CODE_DIR/outputs/data/nell23k/recurrent_relation_prediction.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$VERSION_DIR/results/runs/${RUN_ID}_nell23k_qwen05b_smoke}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "error: RUN_ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "error: run configs/setup_wsl3090.sh first" >&2
  exit 1
fi
if [[ ! -f "$NELL_INPUT" ]]; then
  echo "error: NELL23K recurrent input not found: $NELL_INPUT" >&2
  echo "run: bash configs/prepare_nell23k.sh" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "error: run directory already exists; choose a new RUN_ID: $RUN_DIR" >&2
  exit 1
fi
command -v timeout >/dev/null 2>&1 || { echo "error: GNU timeout is required in WSL2" >&2; exit 1; }

mkdir -p "$RUN_DIR"
cp "$NELL_INPUT.stats.json" "$RUN_DIR/input_stats.json" 2>/dev/null || true
{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "dataset=NELL23K"
  echo "nell_input=$NELL_INPUT"
  date --iso-8601=seconds
  nvidia-smi
  "$PYTHON" "$VERSION_DIR/configs/verify_wsl3090.py"
  "$PYTHON" -m pip freeze 2>/dev/null || uv pip freeze --python "$PYTHON"
} > "$RUN_DIR/environment.txt" 2>&1

cd "$CODE_DIR"
timeout --signal=TERM --kill-after=5m 45m \
  "$PYTHON" scripts/run_recurrent_pilot.py \
    --input_file "$NELL_INPUT" \
    --output_file "$RUN_DIR/predictions.jsonl" \
    --training_output_dir "$RUN_DIR/trainer" \
    --model_name qwen-0.5b \
    --max_graphs 1 \
    --involve_qa_epochs 1 \
    --gen_max_length 24 \
    --wall_time_limit_minutes 30 \
    --hard_stop_minutes 45 \
    --seed 2026 \
    -- \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --bf16 true \
    --recurrent_depth_train 2 \
    --recurrent_depth_sweep 1 2 \
    --target_modules q_proj k_proj v_proj \
    --adapter_control all \
    --evaluation_device cuda \
    --require_cuda true \
    --adapter_output_dir "$RUN_DIR/adapters" \
  2>&1 | tee "$RUN_DIR/run.log"

echo "$RUN_DIR" > "$VERSION_DIR/results/LAST_NELL23K_SMOKE_RUN.txt"
echo "NELL23K smoke completed: $RUN_DIR"

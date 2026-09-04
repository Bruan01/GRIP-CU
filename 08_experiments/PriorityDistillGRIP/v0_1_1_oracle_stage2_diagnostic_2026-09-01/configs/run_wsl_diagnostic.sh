#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}"
RUN_ID="${RUN_ID:-wsl3090_oracle_stage2_diagnostic_$(date +%Y%m%d_%H%M%S)}"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results/logs
bash configs/check_wsl_runtime.sh
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_ROOT/results/runs/$RUN_ID}"
LOG_FILE="results/logs/${RUN_ID}.log"
printf 'run_id=%s\nmodel=%s\noutput=%s\nlog=%s\n' "$RUN_ID" "$MODEL_NAME_OR_PATH" "$OUTPUT_DIR" "$LOG_FILE"
"$PYTHON_EXECUTABLE" scripts/run_diagnostic.py --config configs/oracle_stage2_diagnostic.json --output-dir "$OUTPUT_DIR" --model-name-or-path "$MODEL_NAME_OR_PATH" --overwrite 2>&1 | tee "$LOG_FILE"
"$PYTHON_EXECUTABLE" scripts/summarize_diagnostic.py --run-dir "$OUTPUT_DIR"

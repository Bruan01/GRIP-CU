#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
RUN_ID="${RUN_ID:-wsl3090_oracle_prefix_full_$(date +%Y%m%d_%H%M%S)}"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results/logs

bash configs/check_wsl_runtime.sh
log_file="results/logs/${RUN_ID}.log"
printf 'run_id=%s\nmodel=%s\nlog=%s\n' "$RUN_ID" "$MODEL_NAME_OR_PATH" "$log_file"
"$PYTHON_EXECUTABLE" scripts/run_suite.py \
    --run-id "$RUN_ID" \
    --full-seeds \
    --model-name-or-path "$MODEL_NAME_OR_PATH" \
    --python-executable "$PYTHON_EXECUTABLE" \
    --resume \
    2>&1 | tee "$log_file"

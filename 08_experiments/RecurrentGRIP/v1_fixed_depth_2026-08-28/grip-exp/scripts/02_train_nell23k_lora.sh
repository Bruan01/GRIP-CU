#!/usr/bin/env bash
# Linux equivalent of 02_train_nell23k_lora.ps1.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
    if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
        PYTHON_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python"
    else
        PYTHON_EXECUTABLE="python3"
    fi
else
    PYTHON_EXECUTABLE="$PYTHON_EXECUTABLE"
fi
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: scripts/02_train_nell23k_lora.sh [options]

Options:
  --dry-run                 Print the command without starting Python.
  --python-executable PATH  Python interpreter (default: $PYTHON_EXECUTABLE).
  -h, --help                Show this help message.
USAGE
}

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --python-executable)
            (($# >= 2)) || { echo "error: --python-executable requires a value" >&2; exit 2; }
            PYTHON_EXECUTABLE="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd -- "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
mkdir -p artifacts/logs

log_file="$PROJECT_ROOT/artifacts/logs/02_train_lora_$(date +%Y%m%d_%H%M%S).log"
args=(
    scripts/train_nell23k_lora.py
    --task-file artifacts/nell23k_paper/tasks.json
    --adapter-dir artifacts/nell23k_paper/lora_adapter
    --trainer-output-dir artifacts/nell23k_paper/trainer
    --seed 2026
    --model-name qwen-7b
    --model-source local
    --model-cache-dir model_cache
    --local-files-only
    --lora-r 4
    --lora-alpha 8
    --target-modules down_proj up_proj gate_proj
    --num-train-epochs 1
    --involve-qa-epochs 10
    --s1-stop-loss-threshold 0.15
    --s2-stop-loss-threshold 0.15
    --s1-min-epoch 1
    --s2-min-epoch 1
    --per-device-train-batch-size 1
    --gradient-accumulation-steps 512
    --learning-rate 1e-3
    --weight-decay 1e-4
    --adam-beta1 0.9
    --adam-beta2 0.98
    --adam-epsilon 1e-8
    --max-grad-norm 1.0
)

printf 'Stage 2 only: train and save LoRA.\nLog: %s\n' "$log_file"
printf 'Command: %q' "$PYTHON_EXECUTABLE"
printf ' %q' "${args[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi

"$PYTHON_EXECUTABLE" "${args[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
if ((status != 0)); then
    echo "Stage 2 failed with exit code $status. Log: $log_file" >&2
    exit "$status"
fi

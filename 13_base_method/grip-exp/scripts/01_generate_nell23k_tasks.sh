#!/usr/bin/env bash
# Linux equivalent of 01_generate_nell23k_tasks.ps1.
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
Usage: scripts/01_generate_nell23k_tasks.sh [options]

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

log_file="$PROJECT_ROOT/artifacts/logs/01_generate_tasks_$(date +%Y%m%d_%H%M%S).log"
args=(
    scripts/generate_nell23k_tasks.py
    --input-file outputs/data/nell23k/processed_test.json
    --ref-file outputs/data/nell23k/processed_val.json
    --task-file artifacts/nell23k_paper/tasks.json
    --task-cache-dir artifacts/task_cache/nell23k_paper_qwen
    --seed 2026
    --model-name qwen-7b
    --model-source local
    --model-cache-dir model_cache
    --local-files-only
    --task-generator-model-name qwen-7b
    --no-task-generator-use-vllm
    --task-generator-batch-size 4
    --num-context-qa 8000
    --num-reason-qa 2000
    --num-summarization 6000
    --no-sample-node-attribute-task
    --no-repharse-context-qa
    --no-context-upsampling
    --format-as-instruction
    --task-gen-max-length 1000
    --tokenize-max-length 4096
    --involve-qa-epochs 10
)

printf 'Stage 1 only: create/cached GRIP training tasks.\nLog: %s\n' "$log_file"
printf 'Resume cache: artifacts/task_cache/nell23k_paper_qwen\n'
printf 'Command: %q' "$PYTHON_EXECUTABLE"
printf ' %q' "${args[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi

"$PYTHON_EXECUTABLE" "${args[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
if ((status != 0)); then
    echo "Stage 1 stopped with exit code $status. Rerun this script: cached generation will resume." >&2
    exit "$status"
fi

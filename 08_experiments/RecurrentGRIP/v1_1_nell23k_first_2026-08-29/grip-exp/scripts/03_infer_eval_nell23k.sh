#!/usr/bin/env bash
# Linux equivalent of 03_infer_eval_nell23k.ps1.
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
RESUME=0

usage() {
    cat <<'USAGE'
Usage: scripts/03_infer_eval_nell23k.sh [options]

Options:
  --resume                  Continue an existing partial prediction file.
  --dry-run                 Print the command without starting Python.
  --python-executable PATH  Python interpreter (default: $PYTHON_EXECUTABLE).
  -h, --help                Show this help message.
USAGE
}

while (($#)); do
    case "$1" in
        --resume) RESUME=1 ;;
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

log_file="$PROJECT_ROOT/artifacts/logs/03_infer_eval_$(date +%Y%m%d_%H%M%S).log"
args=(
    scripts/infer_eval_nell23k.py
    --input-file outputs/data/nell23k/processed_test.json
    --adapter-dir artifacts/nell23k_paper/lora_adapter
    --output-file outputs/grip_inf/nell23k/nell23k_qwen_paper_3090.jsonl
    --seed 2026
    --model-name qwen-7b
    --model-source local
    --model-cache-dir model_cache
    --local-files-only
    --tokenize-max-length 4096
    --gen-max-length 1000
)
if ((RESUME)); then
    args+=(--resume)
fi

printf 'Stage 3 only: infer and evaluate.\nLog: %s\n' "$log_file"
printf 'Command: %q' "$PYTHON_EXECUTABLE"
printf ' %q' "${args[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi

"$PYTHON_EXECUTABLE" "${args[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
if ((status != 0)); then
    echo "Stage 3 failed with exit code $status. Rerun with --resume to continue a partial JSONL prediction file." >&2
    exit "$status"
fi

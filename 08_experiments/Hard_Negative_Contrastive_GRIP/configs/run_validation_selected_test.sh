#!/usr/bin/env bash
# Test-only generation using the finalized validation winner; resumes prediction JSONL.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
export SEARCH_DIR="${SEARCH_DIR:-$HNG/results/runs/paper_v2_nell23k/hpo_budget60_seed2026}"
export EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
export TEST_DIR="${TEST_DIR:-$SEARCH_DIR/selected_test}"
export TMUX_SESSION="${TMUX_SESSION:-nell-selected-test}"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ARGS=(--search_dir "$SEARCH_DIR" --eval_file "$EVAL_FILE" --output_dir "$TEST_DIR"
      --model_cache_dir "$CODE_DIR/model_cache")
cd "$CODE_DIR"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec "$PYTHON" "$HNG/scripts/eval_validation_selected.py" "${ARGS[@]}" --dry_run
fi
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
mkdir -p "$TEST_DIR"
exec > >(tee -a "$TEST_DIR/run.log") 2>&1
echo "[test] session=$TMUX_SESSION output=$TEST_DIR started=$(date --iso-8601=seconds)"
"$PYTHON" "$HNG/scripts/eval_validation_selected.py" "${ARGS[@]}"
echo "[test] completed=$(date --iso-8601=seconds)"

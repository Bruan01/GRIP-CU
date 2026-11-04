#!/usr/bin/env bash
# CPU plan: DATASET=codexm SEED=2026 DRY_RUN=1 bash configs/run_ccfb_sprint.sh
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
export DATASET="${DATASET:?set DATASET=nell23k/fb15k237/wn18rr/codexm}"
export SEED="${SEED:?set the declared S2 seed}"
export TMUX_SESSION="${TMUX_SESSION:-ccfb-$DATASET-$SEED}"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ARGS=(--dataset "$DATASET" --seed "$SEED")
[[ -z "${ROOT:-}" ]] || ARGS+=(--root "$ROOT")
[[ -z "${SHARED_ROOT:-}" ]] || ARGS+=(--shared_root "$SHARED_ROOT")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec "$PYTHON" "$HNG/scripts/run_ccfb_sprint.py" "${ARGS[@]}" --dry_run
fi
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
cd "$CODE_DIR"
"$PYTHON" "$HNG/scripts/run_ccfb_sprint.py" "${ARGS[@]}"

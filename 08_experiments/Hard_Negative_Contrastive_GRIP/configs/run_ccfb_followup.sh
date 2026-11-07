#!/usr/bin/env bash
# Launch a prepared follow-up after checking its measured checkpoint budget.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
export DATASET="${DATASET:?set the follow-up dataset}"
export SEED="${SEED:?set the declared S2 seed}"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Resource measurements and prepared manifests refer to the canonical output root.
if [[ -n "${ROOT:-}" || -n "${SHARED_ROOT:-}" ]]; then
  echo 'Use the canonical follow-up directories; ROOT/SHARED_ROOT overrides are unsupported here.' >&2
  exit 2
fi
if [[ "${CCFB_NELL_COMPACT:-0}" == 1 ]]; then
  if [[ "$DATASET:$SEED" != nell23k:2026 ]]; then
    echo 'CCFB_NELL_COMPACT=1 is for the matched NELL23K seed2026 B1 only.' >&2
    exit 2
  fi
  export TMUX_SESSION="${TMUX_SESSION:-ccfb-nell23k-2026}"
  export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exec "$PYTHON" "$HNG/scripts/run_ccfb_nell_b1_compact.py" --dry_run
  fi
  source "$HNG/configs/tmux_guard.sh"
  tmux_guard_reexec "$0" "$@"
  exec "$PYTHON" "$HNG/scripts/run_ccfb_nell_b1_compact.py"
fi
"$PYTHON" "$HNG/scripts/prepare_ccfb_followups.py" --dataset "$DATASET" --seed "$SEED" --check_launch
exec bash "$HNG/configs/run_ccfb_sprint.sh"

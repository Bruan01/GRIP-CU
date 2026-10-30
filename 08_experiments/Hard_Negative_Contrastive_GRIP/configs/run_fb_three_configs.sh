#!/usr/bin/env bash
# FB15k-237 shared S1 -> 3-epoch B1 -> three frozen variants -> validation selection -> test.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp" && pwd)"
export PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
export ROOT="${ROOT:-$HNG/results/runs/fb15k237_three_configs_seed2026}"
export TMUX_SESSION="${TMUX_SESSION:-fb-three-configs-2026}"
export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec "$PYTHON" "$HNG/scripts/run_fb_three_configs.py" --root "$ROOT" --dry_run
fi
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1
echo "[fb] session=$TMUX_SESSION root=$ROOT started=$(date --iso-8601=seconds)"
"$PYTHON" "$HNG/scripts/run_fb_three_configs.py" --root "$ROOT"

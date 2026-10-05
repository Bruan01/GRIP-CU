#!/usr/bin/env bash
# Closed-set 10-way scoring for frozen shared-pool listed adapters.
#
# Inference only. Does not train. Reuses frozen B1 closed-set files already
# copied into each listed run. Default is the 96-question aligned smoke split.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction.aligned.json}"
PROGRESS_EVERY="${PROGRESS_EVERY:-16}"
TMUX_SESSION="${TMUX_SESSION:-shared-pool-closed-set-20260929}"
LAST_RUN_FILE="$HNG/results/LAST_SHARED_POOL_CLOSED_SET_RUN.txt"

RANDOM_K_DIR="${RANDOM_K_DIR:-$HNG/results/runs/20260926_shared_pool_random_k_full}"
TOP_K_DIR="${TOP_K_DIR:-$HNG/results/runs/20260926_shared_pool_top_k_hard_full}"
ADAPTIVE_DIR="${ADAPTIVE_DIR:-$HNG/results/runs/20260926_shared_pool_coverage_adaptive_k_full}"
SOFT_DIR="${SOFT_DIR:-$HNG/results/runs/20261005_shared_pool_truncated_soft_mix_full}"
CALIBRATED_DIR="${CALIBRATED_DIR:-$HNG/results/runs/20261005_shared_pool_truncated_calibrated_full}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'; then
  echo "error: PyTorch cannot see a CUDA device; refusing to run CPU experiment" >&2
  exit 1
fi

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

wait_for_gpu_holder() {
  local pattern="${WAIT_FOR_PATTERN:-}"
  if [[ -z "$pattern" ]]; then
    return 0
  fi
  echo "[wait] until no python cmdline contains: $pattern"
  while pgrep -af python | grep -F "$pattern" | grep -v grep >/dev/null; do
    echo "[wait] still occupied $(date --iso-8601=seconds)"
    sleep 60
  done
  echo "[wait] clear $(date --iso-8601=seconds)"
}

wait_for_gpu_holder

export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

decode_one() {
  local run_dir="$1"
  local adapter="$run_dir/listed/adapter"
  local closed="$run_dir/listed/summary_closed_set.json"
  if [[ ! -d "$adapter" ]]; then
    echo "[closed_set] skip missing adapter $adapter"
    return 0
  fi
  if [[ -f "$closed" && "${FORCE_CLOSED:-0}" != "1" ]]; then
    echo "[closed_set] reuse $closed"
  else
    echo "[closed_set] decode $adapter"
    "$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
      --adapter_dir "$adapter" \
      --output_dir "$run_dir/listed" \
      --eval_file "$EVAL_FILE" \
      --model_name qwen-7b \
      --model_cache_dir "$CODE_DIR/model_cache" \
      --decode closed_set \
      --gen_max_length 32 \
      --progress_every "$PROGRESS_EVERY" \
      --eval_name smoke
  fi
  if [[ -f "$run_dir/b1/summary.json" && -f "$run_dir/listed/summary.json" ]]; then
    "$PYTHON" "$HNG/scripts/eval_listed_decode.py" \
      --compare \
      --eval_file "$EVAL_FILE" \
      --output_dir "$run_dir" \
      --eval_name smoke
  fi
  bash "$HNG/configs/archive_and_push_results.sh" "$run_dir"
}

echo "[closed_set] start $(date --iso-8601=seconds)"
for run_dir in "$RANDOM_K_DIR" "$TOP_K_DIR" "$ADAPTIVE_DIR" "$SOFT_DIR" "$CALIBRATED_DIR"; do
  decode_one "$run_dir"
done
echo "$SOFT_DIR" > "$LAST_RUN_FILE"
echo "[closed_set] compare five arms $(date --iso-8601=seconds)"
SKIP_TMUX=1 bash "$HNG/configs/run_shared_pool_ablation_compare.sh"
echo "[closed_set] done $(date --iso-8601=seconds)"

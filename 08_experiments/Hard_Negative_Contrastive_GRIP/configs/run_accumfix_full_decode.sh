#!/usr/bin/env bash
# Evaluate the fixed Random-K/Calibrated pair sequentially; never retrain.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/20261011_accumfix_full_decode}"
B1_DECODE_RUN="${B1_DECODE_RUN:-$HNG/results/runs/20260915_140500_qwen7b_full_decode}"
TMUX_SESSION="${TMUX_SESSION:-accumfix-full-decode-20261011}"
export PYTHON RUN_DIR B1_DECODE_RUN TMUX_SESSION
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
trap 'echo "[queue] failed at line $LINENO; rerun with the same RUN_DIR to resume"' ERR
printf '[queue] started %s\n' "$(date -u --iso-8601=seconds)"
"$PYTHON" "$HNG/scripts/compare_accumfix_decode.py" \
  --output-dir "$RUN_DIR" --b1-run "$B1_DECODE_RUN" --preflight
"$PYTHON" -c 'import torch; torch.cuda.init(); print("CUDA", torch.cuda.get_device_name(0), "free/total", torch.cuda.mem_get_info(), flush=True)'
printf '%s\n' 'B1 predates the accumulation fix and is a historical reference only. Primary comparison: Calibrated vs Random-K, test generation EM.' > "$RUN_DIR/BASELINE_CAVEAT.txt"
for variant in random_k calibrated; do
  echo "[queue] starting $variant full generation + closed-set evaluation"
  SCALE=full DECODE=both NO_RESUME=0 RESUME_LAST=0 \
    EVAL_FILE="$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json" \
    LISTED_ADAPTER="$HNG/results/runs/20261007_accumfix_${variant}_full/listed/adapter" \
    ADAPTER_RUN="$HNG/results/runs/20261007_accumfix_${variant}_full" \
    RUN_DIR="$RUN_DIR/$variant" SKIP_RESULT_ARCHIVE=1 \
    bash "$HNG/configs/run_listed_only_decode.sh"
done
"$PYTHON" "$HNG/scripts/compare_accumfix_decode.py" \
  --output-dir "$RUN_DIR" --b1-run "$B1_DECODE_RUN"
for variant in random_k calibrated; do
  bash "$HNG/configs/archive_and_push_results.sh" "$RUN_DIR/$variant"
done
printf '[queue] complete %s; paired comparison: %s/paired_comparison.json\n' "$(date -u --iso-8601=seconds)" "$RUN_DIR"

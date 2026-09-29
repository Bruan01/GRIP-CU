#!/usr/bin/env bash
# Sequential listed training for Prompt-6 Soft-Mix then Calibrated.
#
# Same 20260913 Stage-1 adapter, frozen B1, and experiment-H budget
# (~12014 QA, accum=512, ~230 steps). Soft-Mix runs first. Calibrated starts
# only after Soft-Mix finishes with a listed/summary.json. One GPU at a time.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

SAMPLER_DIR="${SAMPLER_DIR:-$HNG/results/runs/20260929_shared_pool_mixture_samplers}"
EXPECTED_LISTED="${EXPECTED_LISTED:-3253}"
SOFT_RUN_DIR="${SOFT_RUN_DIR:-$HNG/results/runs/20260929_shared_pool_soft_mix_full}"
CALIBRATED_RUN_DIR="${CALIBRATED_RUN_DIR:-$HNG/results/runs/20260929_shared_pool_calibrated_full}"

if [[ ! -f "$SAMPLER_DIR/soft_mix.jsonl" || ! -f "$SAMPLER_DIR/calibrated.jsonl" ]]; then
  echo "error: missing mixture manifests under $SAMPLER_DIR" >&2
  echo "  freeze first: bash $HNG/configs/run_freeze_mixture_samplers.sh" >&2
  exit 1
fi

log() {
  echo "[mixture] $*"
}

log "CPU-wire frozen Soft-Mix / Calibrated onto the paper task file"
OUTPUT_DIR="${SAMPLER_DIR}/wired_full" SAMPLER_DIR="$SAMPLER_DIR" \
  EXPECTED_LISTED="$EXPECTED_LISTED" VARIANTS="soft_mix calibrated" SKIP_TMUX=1 \
  bash "$HNG/configs/run_wire_shared_pool_samplers.sh"

log "train Soft-Mix"
ALLOW_NON_RANDOM=1 VARIANT=soft_mix \
  SAMPLER_DIR="$SAMPLER_DIR" \
  RUN_DIR="$SOFT_RUN_DIR" \
  RUN_ID=20260929_shared_pool_soft_mix_full \
  TMUX_SESSION="${TMUX_SESSION:-shared-pool-mixture-full-20260929}" \
  SKIP_TMUX=1 \
  bash "$HNG/configs/run_shared_pool_random_k_full.sh"

if [[ ! -f "$SOFT_RUN_DIR/listed/summary.json" ]]; then
  echo "error: Soft-Mix finished without listed/summary.json: $SOFT_RUN_DIR" >&2
  exit 1
fi

log "train Calibrated after Soft-Mix"
ALLOW_NON_RANDOM=1 VARIANT=calibrated \
  SAMPLER_DIR="$SAMPLER_DIR" \
  RUN_DIR="$CALIBRATED_RUN_DIR" \
  RUN_ID=20260929_shared_pool_calibrated_full \
  TMUX_SESSION="${TMUX_SESSION:-shared-pool-mixture-full-20260929}" \
  SKIP_TMUX=1 \
  bash "$HNG/configs/run_shared_pool_random_k_full.sh"

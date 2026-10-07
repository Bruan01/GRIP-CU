#!/usr/bin/env bash
# Retrain Random-K then Calibrated after the accumulation-normalization fix.
#
# Same 20260913 Stage-1 adapter, frozen manifests, batch, learning rate, and
# lambda. New run directories only: do not resume a pre-fix Stage-2 checkpoint.
# Soft-Mix is not part of this pair. One GPU, Random-K first.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

RANDOM_SAMPLER_DIR="${RANDOM_SAMPLER_DIR:-$HNG/results/runs/20260926_shared_pool_samplers}"
CALIBRATED_SAMPLER_DIR="${CALIBRATED_SAMPLER_DIR:-$HNG/results/runs/20261005_shared_pool_truncated_mixture_samplers}"
RANDOM_RUN_DIR="${RANDOM_RUN_DIR:-$HNG/results/runs/20261007_accumfix_random_k_full}"
CALIBRATED_RUN_DIR="${CALIBRATED_RUN_DIR:-$HNG/results/runs/20261007_accumfix_calibrated_full}"

for path in \
  "$RANDOM_SAMPLER_DIR/random_k.jsonl" \
  "$CALIBRATED_SAMPLER_DIR/calibrated.jsonl" \
  "$CALIBRATED_SAMPLER_DIR/policy.json"
do
  if [[ ! -f "$path" ]]; then
    echo "error: missing required path: $path" >&2
    exit 1
  fi
done
if [[ -e "$RANDOM_RUN_DIR/listed/adapter" || -e "$CALIBRATED_RUN_DIR/listed/adapter" ]]; then
  echo "error: an accumfix adapter already exists; refusing to mix runs" >&2
  echo "  random_k=$RANDOM_RUN_DIR" >&2
  echo "  calibrated=$CALIBRATED_RUN_DIR" >&2
  exit 1
fi

log() {
  echo "[accumfix] $*"
}

log "train Random-K from the shared Stage-1 adapter"
ALLOW_NON_RANDOM=1 VARIANT=random_k \
  SAMPLER_DIR="$RANDOM_SAMPLER_DIR" \
  RUN_DIR="$RANDOM_RUN_DIR" \
  RUN_ID=20261007_accumfix_random_k_full \
  NO_RESUME=1 \
  TMUX_SESSION="${TMUX_SESSION:-accumfix-rk-cal-20261007}" \
  SKIP_TMUX=1 \
  bash "$HNG/configs/run_shared_pool_random_k_full.sh"

if [[ ! -f "$RANDOM_RUN_DIR/listed/summary.json" ]]; then
  echo "error: Random-K finished without listed/summary.json: $RANDOM_RUN_DIR" >&2
  exit 1
fi

log "train Calibrated from the same Stage-1 adapter"
ALLOW_NON_RANDOM=1 VARIANT=calibrated \
  SAMPLER_DIR="$CALIBRATED_SAMPLER_DIR" \
  RUN_DIR="$CALIBRATED_RUN_DIR" \
  RUN_ID=20261007_accumfix_calibrated_full \
  NO_RESUME=1 \
  TMUX_SESSION="${TMUX_SESSION:-accumfix-rk-cal-20261007}" \
  SKIP_TMUX=1 \
  bash "$HNG/configs/run_shared_pool_random_k_full.sh"

if [[ ! -f "$CALIBRATED_RUN_DIR/listed/summary.json" ]]; then
  echo "error: Calibrated finished without listed/summary.json: $CALIBRATED_RUN_DIR" >&2
  exit 1
fi

log "Random-K and Calibrated finished"

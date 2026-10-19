#!/usr/bin/env bash
# Keep the newest trainer checkpoint of the running H5 seed queue.
#
# The H5 queue process (started 2026-10-18) loaded the old prune_checkpoints(),
# which deletes trainer_<variant>/ once the final adapter exists. This watcher
# hardlinks the newest checkpoint into trainer_<variant>.kept/ every minute, so
# the rm only drops a link and the last checkpoint survives at no extra disk.
# Exits when the H5 queue lock is released.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$HNG/results/runs"
TMUX_SESSION="${TMUX_SESSION:-h5-keep-ckpt-20261019}"
H5_QUEUE_DIR="${H5_QUEUE_DIR:-$RUNS/20261018_h5_seed_queue}"
TRAINERS=(
  "$RUNS/20261018_nell_seed2027/train_listed/trainer_listed"
  "$RUNS/20261018_fb15k237_seed2027/train_b1/trainer_b1"
  "$RUNS/20261018_fb15k237_seed2027/train_listed/trainer_listed"
)
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"
log() { echo "[keep-ckpt] $* $(date -u --iso-8601=seconds)"; }

mirror() {  # <trainer_dir>
  local trainer="$1" kept="$1.kept" last name
  [[ -d "$trainer" ]] || return 0
  last=$(find "$trainer" -maxdepth 1 -type d -name 'checkpoint-*' | sort -t- -k2 -n | tail -1)
  [[ -n "$last" && -f "$last/trainer_state.json" ]] || return 0
  name=$(basename "$last")
  [[ -d "$kept/$name" ]] && return 0
  mkdir -p "$kept"
  cp -al "$last" "$kept/$name.tmp" && mv "$kept/$name.tmp" "$kept/$name"
  find "$kept" -maxdepth 1 -mindepth 1 -type d ! -name "$name" -exec rm -rf -- {} +
  log "mirrored $last -> $kept/$name"
}

log "watching ${#TRAINERS[@]} trainer dirs"
while true; do
  for t in "${TRAINERS[@]}"; do mirror "$t" || true; done
  # The FB Calibrated queue takes the same lock afterwards, so watch the session.
  if ! tmux has-session -t "${H5_SESSION:-h5-seeds-20261018}" 2>/dev/null; then
    for t in "${TRAINERS[@]}"; do mirror "$t" || true; done
    log "H5 session gone; exiting"; break
  fi
  sleep 60
done

#!/usr/bin/env bash
set -euo pipefail

# Reproducible V100 profile.  It preserves the full NELL23K training budget
# and changes only hardware-safe precision plus throughput/I/O defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HNG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DATASET="${DATASET:-nell23k}"
export NEGATIVE_SOURCE="${NEGATIVE_SOURCE:-structure_counterfactual}"
export TORCH_DTYPE="${TORCH_DTYPE:-float16}"
export GRIP_ATTENTION_IMPLEMENTATION="${GRIP_ATTENTION_IMPLEMENTATION:-sdpa}"
export CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-16}"
export SAVE_STEPS="${SAVE_STEPS:-24}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

exec bash "$HNG_ROOT/configs/run_structure_counterfactual_wsl.sh" "$@"

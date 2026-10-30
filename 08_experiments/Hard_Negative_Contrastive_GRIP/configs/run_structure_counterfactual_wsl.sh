#!/usr/bin/env bash
# Canonical entrypoint for structure-counterfactual K sweeps + InfoNCE.
# The implementation is shared with the legacy listed-negative runner so that
# the ablation and mainline keep identical training/evaluation contracts.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEGATIVE_SOURCE="${NEGATIVE_SOURCE:-structure_counterfactual}"
export NEGATIVE_SOURCE
exec bash "$HNG/configs/run_error_memory_wsl.sh" "$@"

#!/usr/bin/env bash
# Strict-paper GRIP B1 on NELL23K only (tasks -> S1 -> S2 B1 per seed). Not run yet.
# Thin wrapper: every paper setting lives in run_paper_pipeline.sh. Seeds via SEEDS="2026 2027".
set -euo pipefail
export DATASET=nell23k STEPS="${STEPS:-tasks s1 b1}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_paper_pipeline.sh" "$@"

#!/usr/bin/env bash
# Wait for a listed-only training PID, then write listed-vs-frozen-B1 comparison.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_PID="${WAIT_PID:?set WAIT_PID to the training python process}"
LISTED_RUN="${LISTED_RUN:-$HNG/results/runs/20260917_qwen7b_train_graph_negatives}"
B1_RUN="${B1_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
POLL_SECONDS="${POLL_SECONDS:-120}"

echo "[wait] pid=$WAIT_PID run=$LISTED_RUN"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

if [[ ! -f "$LISTED_RUN/listed/summary.json" ]]; then
  echo "error: pid $WAIT_PID exited without $LISTED_RUN/listed/summary.json" >&2
  exit 1
fi

LISTED_RUN="$LISTED_RUN" B1_RUN="$B1_RUN" bash "$HNG/configs/compare_listed_to_frozen_b1.sh"
echo "[wait] comparison ready $(date --iso-8601=seconds)"

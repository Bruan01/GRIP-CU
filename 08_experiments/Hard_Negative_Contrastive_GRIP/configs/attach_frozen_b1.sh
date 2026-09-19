#!/usr/bin/env bash
# Copy the frozen 20260913 B1 smoke summary into a listed-only run directory.
# Lambda=0 never used InfoNCE negatives, so B1 is not retrained.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LISTED_RUN="${LISTED_RUN:?set LISTED_RUN to the listed-only run directory}"
B1_RUN="${B1_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"

if [[ ! -f "$B1_RUN/b1/summary.json" ]]; then
  echo "error: missing frozen B1 summary: $B1_RUN/b1/summary.json" >&2
  exit 1
fi
if [[ ! -d "$LISTED_RUN" ]]; then
  echo "error: listed run directory does not exist: $LISTED_RUN" >&2
  exit 1
fi

mkdir -p "$LISTED_RUN/b1"
cp -f "$B1_RUN/b1/summary.json" "$LISTED_RUN/b1/summary.json"
if [[ -f "$B1_RUN/b1/predictions_correct.jsonl" ]]; then
  cp -f "$B1_RUN/b1/predictions_correct.jsonl" "$LISTED_RUN/b1/predictions_correct.jsonl"
fi
if [[ -f "$B1_RUN/b1/summary_closed_set.json" ]]; then
  cp -f "$B1_RUN/b1/summary_closed_set.json" "$LISTED_RUN/b1/summary_closed_set.json"
fi
if [[ -f "$B1_RUN/b1/predictions_closed_set.jsonl" ]]; then
  cp -f "$B1_RUN/b1/predictions_closed_set.jsonl" "$LISTED_RUN/b1/predictions_closed_set.jsonl"
fi
cat > "$LISTED_RUN/b1/FROZEN_FROM.json" <<EOF
{
  "source": "$B1_RUN",
  "copied_at_utc": "$(date -u --iso-8601=seconds)",
  "note": "Frozen generation-only B1. lambda=0 never used listed negatives."
}
EOF
echo "[attach] froze B1 summaries from $B1_RUN into $LISTED_RUN/b1"

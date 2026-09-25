#!/usr/bin/env bash
# CPU-only quality audit of the production confusion dump.
# Does not load the 7B teacher and does not train.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
TASK_FILE="${TASK_FILE:-$HNG/grip_nell23k_tasks.json}"
SCORES="${SCORES:-$HNG/results/runs/20260923_offline_confusion_full/candidate_scores.jsonl}"
QA_SUMMARY="${QA_SUMMARY:-$HNG/results/runs/20260923_offline_confusion_full/qa_summary.jsonl}"
METADATA="${METADATA:-$HNG/results/runs/20260923_offline_confusion_full/metadata.json}"
FROZEN_DB="${FROZEN_DB:-$HNG/results/runs/20260923_confusion_db_frozen_b1/confusion_db.jsonl}"
COMPARE_SCORES="${COMPARE_SCORES:-$HNG/results/runs/20260923_offline_confusion_compare20_b8/candidate_scores.jsonl}"
LIVE_RESCORE="${LIVE_RESCORE:-$HNG/results/runs/20260923_offline_confusion_full/audit/live_20qa_rescore.json}"
LISTED_ADAPTER="${LISTED_ADAPTER:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/listed/adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-$HNG/results/runs/20260923_offline_confusion_full/audit}"
REPORT="${REPORT:-$HNG/CONFUSION_DATABASE_AUDIT.md}"
TMUX_SESSION="${TMUX_SESSION:-confusion-audit-cpu-$(date +%Y%m%d)}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
for required in "$SCORES" "$QA_SUMMARY" "$METADATA" "$TASK_FILE" "$RAW_DIR/train.txt"; do
  if [[ ! -e "$required" ]]; then
    echo "error: missing required path: $required" >&2
    exit 1
  fi
done

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
extra=()
if [[ -n "${FROZEN_DB}" && -e "$FROZEN_DB" ]]; then
  extra+=(--frozen_db "$FROZEN_DB")
fi
if [[ -n "${COMPARE_SCORES}" && -e "$COMPARE_SCORES" ]]; then
  extra+=(--compare_scores "$COMPARE_SCORES")
fi
if [[ -n "${LIVE_RESCORE}" && -e "$LIVE_RESCORE" ]]; then
  extra+=(--live_rescore "$LIVE_RESCORE")
fi
if [[ -n "${LISTED_ADAPTER}" ]]; then
  extra+=(--listed_adapter "$LISTED_ADAPTER")
fi
echo "[launch] confusion audit -> $OUTPUT_DIR tmux=${TMUX_SESSION:-none}" | tee -a "$OUTPUT_DIR/cpu_audit.log"
set +e
"$PYTHON" "$HNG/scripts/audit_confusion_db.py" \
  --scores "$SCORES" \
  --qa_summary "$QA_SUMMARY" \
  --metadata "$METADATA" \
  --task_file "$TASK_FILE" \
  --raw_dir "$RAW_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --report "$REPORT" \
  "${extra[@]}" \
  2>&1 | tee -a "$OUTPUT_DIR/cpu_audit.log"
rc=${PIPESTATUS[0]}
set -e
echo "[launch] confusion audit finished exit=$rc $(date --iso-8601=seconds)" | tee -a "$OUTPUT_DIR/cpu_audit.log"
exit "$rc"

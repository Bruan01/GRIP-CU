#!/usr/bin/env bash
# Freeze the existing full-vocab B1 score table as an immutable confusion DB.
# Offline: no GPU, no teacher reload. Set LIMIT=0 mining first if the source
# JSONL is incomplete.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
INPUT_FILE="${INPUT_FILE:-$HNG/grip_nell23k_tasks.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
B1_ADAPTER="${B1_ADAPTER:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke/b1/adapter}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$HNG/results/runs/20260921_031844_score_hard_mining/score_hard_manifest.jsonl}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/20260923_confusion_db_frozen_b1}"
OUTPUT="${OUTPUT:-$RUN_DIR/confusion_db.jsonl}"
METADATA="${METADATA:-$RUN_DIR/confusion_db.json}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-8}"
SEED="${SEED:-2026}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
for required in "$INPUT_FILE" "$RAW_DIR/train.txt" "$SOURCE_MANIFEST" "$B1_ADAPTER"; do
  if [[ ! -e "$required" ]]; then
    echo "error: missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR"
{
  echo "task_file=$INPUT_FILE"
  echo "raw_dir=$RAW_DIR"
  echo "b1_adapter=$B1_ADAPTER"
  echo "source_manifest=$SOURCE_MANIFEST"
  echo "output=$OUTPUT"
  echo "metadata=$METADATA"
  echo "candidate_batch_size=$CANDIDATE_BATCH_SIZE"
  echo "seed=$SEED"
  date --iso-8601=seconds
} >> "$RUN_DIR/environment.txt"

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$HNG/scripts/freeze_confusion_db.py" \
  --task_file "$INPUT_FILE" \
  --raw_dir "$RAW_DIR" \
  --input "$SOURCE_MANIFEST" \
  --output "$OUTPUT" \
  --metadata "$METADATA" \
  --b1_adapter "$B1_ADAPTER" \
  --candidate_batch_size "$CANDIDATE_BATCH_SIZE" \
  --seed "$SEED" \
  | tee -a "$RUN_DIR/run.log"
echo "LAST_CONFUSION_DB=$OUTPUT" > "$HNG/results/LAST_CONFUSION_DB.txt"
echo "$OUTPUT" >> "$HNG/results/LAST_CONFUSION_DB.txt"

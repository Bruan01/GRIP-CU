#!/usr/bin/env bash
# Build official NELL23K val/test questions and rewrite 10-way lists.
#
# Train questions are not used for decode; keep one so the record stays valid.
# Validation/test take every official triple (4950 / 4943).
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
RAW_DIR="${NELL_RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
PREPARED="${PREPARED:-$HNG/data/nell23k/recurrent_relation_prediction_full.json}"
ALIGNED="${ALIGNED:-$HNG/data/nell23k/recurrent_relation_prediction_full.aligned.json}"
REPORT="${REPORT:-$HNG/data/nell23k/official_alignment_report_full.json}"
FORCE="${FORCE:-0}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi

mkdir -p "$(dirname "$PREPARED")" "$(dirname "$ALIGNED")"
if [[ "$FORCE" != "1" && -f "$ALIGNED" ]]; then
  echo "aligned full eval already exists: $ALIGNED"
  exit 0
fi

cd "$CODE_DIR"
echo "[full-eval] prepare $PREPARED"
"$PYTHON" scripts/prepare_recurrent_nell23k.py \
  --raw_dir "$RAW_DIR" \
  --output_file "$PREPARED" \
  --max_train_questions 1 \
  --max_validation_questions 100000 \
  --max_test_questions 100000 \
  --num_candidates 10 \
  --seed 2026

echo "[full-eval] align official 10-way lists -> $ALIGNED"
"$PYTHON" "$HNG/scripts/align_official_nell23k_lists.py" \
  --raw_dir "$RAW_DIR" \
  --input_file "$PREPARED" \
  --output_file "$ALIGNED" \
  --report_file "$REPORT" \
  --seed 2026

echo "prepared full NELL23K eval: $ALIGNED"

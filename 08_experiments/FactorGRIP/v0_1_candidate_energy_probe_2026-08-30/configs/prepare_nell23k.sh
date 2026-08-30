#!/usr/bin/env bash
set -euo pipefail
VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
CONDA_ENV="${CONDA_ENV:-guardenv}"
RAW_DIR="${NELL_RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
OUTPUT_FILE="${NELL_OUTPUT:-$CODE_DIR/outputs/data/nell23k/recurrent_relation_prediction.json}"
MAX_TRAIN_QUESTIONS="${MAX_TRAIN_QUESTIONS:-64}"
MAX_VALIDATION_QUESTIONS="${MAX_VALIDATION_QUESTIONS:-32}"
MAX_TEST_QUESTIONS="${MAX_TEST_QUESTIONS:-64}"
NUM_CANDIDATES="${NUM_CANDIDATES:-10}"
SEED="${SEED:-2026}"
command -v conda >/dev/null 2>&1 || { echo "error: conda is required" >&2; exit 1; }
cd "$CODE_DIR"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/prepare_recurrent_nell23k.py \
  --raw_dir "$RAW_DIR" --output_file "$OUTPUT_FILE" \
  --max_train_questions "$MAX_TRAIN_QUESTIONS" --max_validation_questions "$MAX_VALIDATION_QUESTIONS" \
  --max_test_questions "$MAX_TEST_QUESTIONS" --num_candidates "$NUM_CANDIDATES" --seed "$SEED"
echo "prepared NELL23K recurrent input: $OUTPUT_FILE"

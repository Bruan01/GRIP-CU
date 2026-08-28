#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
RAW_INPUT="${RAW_INPUT:-$CODE_DIR/outputs/data/clegr_reasoning/processed_test.json}"
OUTPUT_FILE="${OUTPUT_FILE:-$CODE_DIR/outputs/data/clegr_reasoning/recurrent_station_shortest.json}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  echo "on WSL2 run configs/setup_wsl3090.sh first" >&2
  exit 1
fi

cd "$CODE_DIR"
"$PYTHON" scripts/prepare_recurrent_clegr.py \
  --input_file "$RAW_INPUT" \
  --output_file "$OUTPUT_FILE" \
  --question_types StationShortestCount \
  --train_hops 1 2 \
  --validation_hops 1 2 \
  --test_hops 3 4 \
  --validation_fraction 0.2 \
  --max_graphs 16 \
  --max_questions_per_hop 32 \
  --seed 2026

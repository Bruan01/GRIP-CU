#!/usr/bin/env bash
# CPU-only: attach frozen shared-pool manifests to the full paper task file.
#
# This is the sampling-protocol gate. It does not load a 7B, does not rescore
# 3253x198, and does not train. The retired 64-QA listed smoke is not a substitute.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
INPUT_FILE="${INPUT_FILE:-$HNG/grip_nell23k_tasks.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
SAMPLER_DIR="${SAMPLER_DIR:-$HNG/results/runs/20260926_shared_pool_samplers}"
OUTPUT_DIR="${OUTPUT_DIR:-$SAMPLER_DIR/wired_full}"
EXPECTED_LISTED="${EXPECTED_LISTED:-3253}"
VARIANTS="${VARIANTS:-}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
required=("$INPUT_FILE" "$RAW_DIR/train.txt" "$SAMPLER_DIR/policy.json")
if [[ -n "$VARIANTS" ]]; then
  read -r -a VARIANT_LIST <<< "$VARIANTS"
else
  VARIANT_LIST=()
  for candidate in random_k top_k_hard coverage_adaptive_k soft_mix calibrated; do
    if [[ -f "$SAMPLER_DIR/${candidate}.jsonl" ]]; then
      VARIANT_LIST+=("$candidate")
    fi
  done
fi
if [[ "${#VARIANT_LIST[@]}" -eq 0 ]]; then
  echo "error: no sampler manifests found under $SAMPLER_DIR" >&2
  exit 1
fi
for variant in "${VARIANT_LIST[@]}"; do
  required+=("$SAMPLER_DIR/${variant}.jsonl")
done
for required_path in "${required[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "error: missing required path: $required_path" >&2
    exit 1
  fi
done

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run.log"

{
  echo "[wire] frozen shared-pool manifests onto $INPUT_FILE"
  echo "sampler_dir=$SAMPLER_DIR"
  echo "expected_listed=$EXPECTED_LISTED"
  echo "note=CPU wiring of the full paper task file; do not train 64-QA / 10-step"
  date --iso-8601=seconds
  "$PYTHON" "$HNG/scripts/verify_shared_pool_sampler_wiring.py" \
    --task_file "$INPUT_FILE" \
    --sampler_dir "$SAMPLER_DIR" \
    --raw_dir "$RAW_DIR" \
    --expected_listed "$EXPECTED_LISTED" \
    --variants "${VARIANT_LIST[@]}" \
    --output "$OUTPUT_DIR/wiring.json"
  echo "[wire] finished $(date --iso-8601=seconds)"
} | tee "$LOG"
bash "$HNG/configs/archive_and_push_results.sh" "$OUTPUT_DIR" | tee -a "$LOG"

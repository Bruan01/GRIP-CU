#!/usr/bin/env bash
# Freeze truncated Soft-Mix / Calibrated from the same train-only dump.
# CPU-only. Does not rescore, does not train, and does not overwrite the
# 20260929 full-pool mixture manifests or the Random-K / Top-K Hard /
# Coverage-Adaptive K control lists.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
SCORES="${SCORES:-$HNG/results/runs/20260923_offline_confusion_train_filter/candidate_scores.jsonl}"
METADATA="${METADATA:-$HNG/results/runs/20260923_offline_confusion_train_filter/metadata.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$HNG/results/runs/20261005_shared_pool_truncated_mixture_samplers}"
K_FIXED="${K_FIXED:-9}"
K_MIN="${K_MIN:-1}"
K_MAX="${K_MAX:-20}"
K_UNIFORM="${K_UNIFORM:-6}"
K_SOFT="${K_SOFT:-3}"
SOFT_POOL_K="${SOFT_POOL_K:-9}"
RHO="${RHO:-0.33}"
LAMBDA_0="${LAMBDA_0:-1.0}"
LAMBDA_MIN="${LAMBDA_MIN:-0.25}"
LAMBDA_BETA="${LAMBDA_BETA:-0.5}"
SEED="${SEED:-2026}"
LOG="${LOG:-$OUTPUT_DIR/run.log}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$SCORES" ]]; then
  echo "error: missing train-only scores: $SCORES" >&2
  exit 1
fi

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
{
  echo "[launch] freeze truncated Soft-Mix / Calibrated -> $OUTPUT_DIR"
  echo "scores=$SCORES"
  echo "metadata=$METADATA"
  echo "k_uniform=$K_UNIFORM k_soft=$K_SOFT soft_pool_k=$SOFT_POOL_K rho=$RHO"
  echo "lambda_0=$LAMBDA_0 lambda_min=$LAMBDA_MIN lambda_beta=$LAMBDA_BETA seed=$SEED"
  "$PYTHON" "$HNG/scripts/freeze_shared_pool_samplers.py" \
    --scores "$SCORES" \
    --metadata "$METADATA" \
    --output_dir "$OUTPUT_DIR" \
    --k_fixed "$K_FIXED" \
    --k_min "$K_MIN" \
    --k_max "$K_MAX" \
    --k_uniform "$K_UNIFORM" \
    --k_soft "$K_SOFT" \
    --soft_pool_k "$SOFT_POOL_K" \
    --rho "$RHO" \
    --lambda_0 "$LAMBDA_0" \
    --lambda_min "$LAMBDA_MIN" \
    --lambda_beta "$LAMBDA_BETA" \
    --seed "$SEED" \
    --variants soft_mix calibrated
  echo "[launch] freeze finished $(date --iso-8601=seconds)"
} | tee "$LOG"
bash "$HNG/configs/archive_and_push_results.sh" "$OUTPUT_DIR" | tee -a "$LOG"

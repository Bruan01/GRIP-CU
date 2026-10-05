#!/usr/bin/env bash
# Sequential listed training for truncated Soft-Mix then Calibrated.
#
# Same 20260913 Stage-1 adapter, frozen B1, and experiment-H budget
# (~12014 QA, accum=512, ~230 steps). Soft-Mix runs first. Calibrated starts
# only after Soft-Mix finishes with a listed/summary.json. After Calibrated
# has listed/summary.json, run closed-set decode on the five listed adapters
# and write the five-arm comparison. One GPU at a time.
# SKIP_CLOSED_SET=1 stops after Calibrated training.
#
# Defaults point at 20261005 Top-N truncated freeze/runs. The 20260929
# full-pool mixture directories are refused unless ALLOW_RETIRED_FULL_POOL=1.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

SAMPLER_DIR="${SAMPLER_DIR:-$HNG/results/runs/20261005_shared_pool_truncated_mixture_samplers}"
EXPECTED_LISTED="${EXPECTED_LISTED:-3253}"
SOFT_RUN_DIR="${SOFT_RUN_DIR:-$HNG/results/runs/20261005_shared_pool_truncated_soft_mix_full}"
CALIBRATED_RUN_DIR="${CALIBRATED_RUN_DIR:-$HNG/results/runs/20261005_shared_pool_truncated_calibrated_full}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
ALLOW_RETIRED_FULL_POOL="${ALLOW_RETIRED_FULL_POOL:-0}" \
HNG="$HNG" SAMPLER_DIR="$SAMPLER_DIR" \
SOFT_RUN_DIR="$SOFT_RUN_DIR" CALIBRATED_RUN_DIR="$CALIBRATED_RUN_DIR" \
"$PYTHON" - <<'PY'
from pathlib import Path
import json
import os
import sys

sys.path.insert(0, str(Path(os.environ["HNG"]) / "src"))
from hard_negative_grip.shared_pool_samplers import (
    assert_truncated_mixture_policy,
    refuse_retired_full_pool_mixture,
)

allow = os.environ.get("ALLOW_RETIRED_FULL_POOL", "0") == "1"
sampler = Path(os.environ["SAMPLER_DIR"])
for path in (
    sampler,
    Path(os.environ["SOFT_RUN_DIR"]),
    Path(os.environ["CALIBRATED_RUN_DIR"]),
):
    refuse_retired_full_pool_mixture(path, allow=allow)
policy_path = sampler / "policy.json"
if policy_path.is_file():
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    assert_truncated_mixture_policy(payload, path=policy_path)
PY

if [[ ! -f "$SAMPLER_DIR/soft_mix.jsonl" || ! -f "$SAMPLER_DIR/calibrated.jsonl" ]]; then
  echo "error: missing truncated mixture manifests under $SAMPLER_DIR" >&2
  echo "  freeze first: bash $HNG/configs/run_freeze_mixture_samplers.sh" >&2
  exit 1
fi

log() {
  echo "[mixture] $*"
}

WIRE_DIR="${SAMPLER_DIR}/wired_full"
if [[ -f "$WIRE_DIR/wiring.json" && "${FORCE_WIRE:-0}" != "1" ]]; then
  log "reuse CPU wiring $WIRE_DIR/wiring.json"
else
  log "CPU-wire frozen Soft-Mix / Calibrated onto the paper task file"
  OUTPUT_DIR="$WIRE_DIR" SAMPLER_DIR="$SAMPLER_DIR" \
    EXPECTED_LISTED="$EXPECTED_LISTED" VARIANTS="soft_mix calibrated" SKIP_TMUX=1 \
    bash "$HNG/configs/run_wire_shared_pool_samplers.sh"
fi

if [[ -f "$SOFT_RUN_DIR/listed/summary.json" && "${FORCE_SOFT_MIX:-0}" != "1" ]]; then
  log "reuse Soft-Mix $SOFT_RUN_DIR/listed/summary.json"
else
  log "train Soft-Mix"
  ALLOW_NON_RANDOM=1 VARIANT=soft_mix \
    SAMPLER_DIR="$SAMPLER_DIR" \
    RUN_DIR="$SOFT_RUN_DIR" \
    RUN_ID=20261005_shared_pool_truncated_soft_mix_full \
    TMUX_SESSION="${TMUX_SESSION:-shared-pool-truncated-mixture-full-20261005}" \
    SKIP_TMUX=1 \
    bash "$HNG/configs/run_shared_pool_random_k_full.sh"
fi

if [[ ! -f "$SOFT_RUN_DIR/listed/summary.json" ]]; then
  echo "error: Soft-Mix finished without listed/summary.json: $SOFT_RUN_DIR" >&2
  exit 1
fi

log "train Calibrated after Soft-Mix"
ALLOW_NON_RANDOM=1 VARIANT=calibrated \
  SAMPLER_DIR="$SAMPLER_DIR" \
  RUN_DIR="$CALIBRATED_RUN_DIR" \
  RUN_ID=20261005_shared_pool_truncated_calibrated_full \
  TMUX_SESSION="${TMUX_SESSION:-shared-pool-truncated-mixture-full-20261005}" \
  SKIP_TMUX=1 \
  bash "$HNG/configs/run_shared_pool_random_k_full.sh"

if [[ ! -f "$CALIBRATED_RUN_DIR/listed/summary.json" ]]; then
  echo "error: Calibrated finished without listed/summary.json: $CALIBRATED_RUN_DIR" >&2
  exit 1
fi

if [[ "${SKIP_CLOSED_SET:-0}" == "1" ]]; then
  log "skip closed-set decode (SKIP_CLOSED_SET=1)"
  exit 0
fi

log "closed-set decode after Calibrated, then five-arm compare"
WAIT_FOR_PATTERN= \
  SOFT_DIR="$SOFT_RUN_DIR" \
  CALIBRATED_DIR="$CALIBRATED_RUN_DIR" \
  TMUX_SESSION="${TMUX_SESSION:-shared-pool-truncated-mixture-full-20261005}" \
  SKIP_TMUX=1 \
  bash "$HNG/configs/run_shared_pool_closed_set.sh"

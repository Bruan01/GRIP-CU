#!/usr/bin/env bash
# Experiment wrapper. The implementation lives in 13_base_method/grip-exp.
set -Eeuo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_ROOT="$(cd -- "$EXPERIMENT_DIR/../../.." && pwd -P)"
GRIP_SCRIPT="$REPO_ROOT/13_base_method/grip-exp/scripts/run_nell23k_qwen05b.sh"

if [[ ! -x "$GRIP_SCRIPT" ]]; then
    echo "Missing executable GRIP baseline entry: $GRIP_SCRIPT" >&2
    exit 1
fi

exec "$GRIP_SCRIPT" "$@"

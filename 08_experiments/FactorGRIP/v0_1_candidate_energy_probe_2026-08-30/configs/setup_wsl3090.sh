#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
CONDA_ENV="${CONDA_ENV:-guardenv}"

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "error: this setup script is intended for WSL2" >&2
  exit 1
fi
command -v conda >/dev/null 2>&1 || { echo "error: conda is required" >&2; exit 1; }
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is unavailable inside WSL2" >&2
  exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "error: conda environment '$CONDA_ENV' does not exist" >&2
  echo "create it first, then install the pinned project dependencies there" >&2
  exit 1
fi

run_python() { conda run --no-capture-output -n "$CONDA_ENV" python "$@"; }
cd "$CODE_DIR"
run_python "$VERSION_DIR/configs/verify_wsl3090.py"
echo "WSL2 RTX 3090 environment ready: conda environment $CONDA_ENV"

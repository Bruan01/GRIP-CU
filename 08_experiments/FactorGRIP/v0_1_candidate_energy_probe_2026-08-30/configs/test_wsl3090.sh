#!/usr/bin/env bash
set -euo pipefail
VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
CONDA_ENV="${CONDA_ENV:-guardenv}"
command -v conda >/dev/null 2>&1 || { echo "error: conda is required" >&2; exit 1; }
cd "$CODE_DIR"
conda run --no-capture-output -n "$CONDA_ENV" python "$VERSION_DIR/configs/verify_wsl3090.py"
conda run --no-capture-output -n "$CONDA_ENV" python -m unittest discover -s tests -v

#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: run configs/setup_wsl3090.sh first" >&2
  exit 1
fi

cd "$CODE_DIR"
"$PYTHON" "$VERSION_DIR/configs/verify_wsl3090.py"
"$PYTHON" -m unittest discover -s tests -v

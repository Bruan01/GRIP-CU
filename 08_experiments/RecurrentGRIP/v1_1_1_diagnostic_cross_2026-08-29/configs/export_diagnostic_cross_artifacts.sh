#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_DIR="$(cd "$VERSION_DIR/../../.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  LAST_RUN_FILE="$VERSION_DIR/results/LAST_NELL23K_DIAGNOSTIC_CROSS_RUN.txt"
  if [[ ! -f "$LAST_RUN_FILE" ]]; then
    echo "error: set RUN_DIR or complete a diagnostic-cross run first" >&2
    exit 1
  fi
  RUN_DIR="$(cat "$LAST_RUN_FILE")"
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
RUN_NAME="$(basename "$RUN_DIR")"
EXPORT_DIR="${EXPORT_DIR:-$REPOSITORY_DIR/09_results_analysis/artifacts/RecurrentGRIP_v1_1_1/$RUN_NAME}"

"$PYTHON" "$VERSION_DIR/configs/export_diagnostic_cross_artifacts.py" \
  --run-dir "$RUN_DIR" \
  --output-dir "$EXPORT_DIR" \
  --repository-dir "$REPOSITORY_DIR"

echo "Audit artifacts exported to: $EXPORT_DIR"
echo "Next: cd '$REPOSITORY_DIR' && git status --short"

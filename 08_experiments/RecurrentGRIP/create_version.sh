#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SOURCE_VERSION NEW_VERSION" >&2
  echo "example: $0 v1_fixed_depth_2026-08-28 v1_1_executor_position_2026-09-05" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$ROOT/$1"
TARGET="$ROOT/$2"

if [[ ! -d "$SOURCE" ]]; then
  echo "source version not found: $SOURCE" >&2
  exit 1
fi
if [[ -e "$TARGET" ]]; then
  echo "target version already exists: $TARGET" >&2
  exit 1
fi

mkdir -p "$TARGET"
rsync -a \
  --exclude='results/*' \
  --exclude='logs/*' \
  --exclude='.venv/' \
  --exclude='model_cache/' \
  --exclude='outputs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  "$SOURCE/" "$TARGET/"
mkdir -p "$TARGET/results" "$TARGET/logs"
touch "$TARGET/results/.gitkeep" "$TARGET/logs/.gitkeep"
echo "created $TARGET"
echo "next: update README.md, SOURCE_BASELINE.md, configs, and the registry before running"

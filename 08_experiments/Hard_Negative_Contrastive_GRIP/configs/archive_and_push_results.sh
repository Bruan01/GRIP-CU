#!/usr/bin/env bash
# Snapshot a finished run into results/archive/<run>/<stamp>/ and git push.
# Never overwrites an existing snapshot. Adapter weights stay out of Git.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HNG/../.." && pwd)"
PYTHON="${PYTHON:-$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi

if [[ "${SKIP_RESULT_ARCHIVE:-0}" == "1" ]]; then
  echo "[archive] SKIP_RESULT_ARCHIVE=1; not snapshotting"
  exit 0
fi

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  echo "usage: bash configs/archive_and_push_results.sh <run_dir>" >&2
  exit 1
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "error: run directory not found: $RUN_DIR" >&2
  exit 1
fi

export PYTHONPATH="$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
DEST="$("$PYTHON" "$HNG/scripts/archive_run_snapshot.py" "$RUN_DIR")"
echo "[archive] snapshot=$DEST"

if [[ "${SKIP_RESULT_PUSH:-0}" == "1" ]]; then
  echo "[archive] SKIP_RESULT_PUSH=1; snapshot kept locally"
  exit 0
fi
if ! command -v git >/dev/null 2>&1; then
  echo "[archive] git not found; snapshot kept locally at $DEST" >&2
  exit 0
fi
if [[ ! -d "$REPO/.git" ]]; then
  echo "[archive] $REPO is not a git repo; snapshot kept locally at $DEST" >&2
  exit 0
fi

INDEX_REL="08_experiments/Hard_Negative_Contrastive_GRIP/results/archive/INDEX.md"
REL_DEST="${DEST#"$REPO/"}"
if [[ "$REL_DEST" == "$DEST" || "$REL_DEST" != 08_experiments/Hard_Negative_Contrastive_GRIP/results/archive/* ]]; then
  echo "[archive] snapshot is outside the Git archive tree; kept locally at $DEST" >&2
  exit 0
fi

BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
git -C "$REPO" add -A -- "$INDEX_REL" "$REL_DEST"
if git -C "$REPO" diff --cached --quiet -- "$INDEX_REL" "$REL_DEST"; then
  echo "[archive] nothing new to commit for $DEST"
  exit 0
fi

RUN_NAME="$(basename "$RUN_DIR")"
STAMP="$(basename "$DEST")"
MESSAGE="chore: archive results ${RUN_NAME} ${STAMP}"
METRICS="$DEST/snapshot_manifest.json"
if [[ -f "$METRICS" ]]; then
  LISTED_EM="$("$PYTHON" -c 'import json,sys; em=(json.load(open(sys.argv[1],encoding="utf-8")).get("metrics") or {}).get("listed_em");
print(f"{100.0*float(em):.2f}%" if isinstance(em,(int,float)) else "")' "$METRICS")"
  B1_EM="$("$PYTHON" -c 'import json,sys; em=(json.load(open(sys.argv[1],encoding="utf-8")).get("metrics") or {}).get("b1_em");
print(f"{100.0*float(em):.2f}%" if isinstance(em,(int,float)) else "")' "$METRICS")"
  if [[ -n "${LISTED_EM:-}" && -n "${B1_EM:-}" ]]; then
    MESSAGE="chore: archive listed EM ${LISTED_EM} vs B1 ${B1_EM} (${RUN_NAME})"
  elif [[ -n "${LISTED_EM:-}" ]]; then
    MESSAGE="chore: archive listed EM ${LISTED_EM} (${RUN_NAME})"
  fi
fi

git -C "$REPO" commit -m "$MESSAGE" -- "$INDEX_REL" "$REL_DEST"
if ! git -C "$REPO" push origin "$BRANCH"; then
  echo "[archive] push failed; pulling rebase then retrying" >&2
  git -C "$REPO" pull --rebase origin "$BRANCH"
  git -C "$REPO" push origin "$BRANCH"
fi
echo "[archive] pushed $MESSAGE on $BRANCH"

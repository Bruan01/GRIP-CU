#!/usr/bin/env bash
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
RUN_ROOT="$HNG/results/hpo/stage2_three_dataset_20261030"
SESSION=grip_stage2_three_datasets
ADOPT_PID="${ADOPT_FB_PID:-0}"
mkdir -p "$RUN_ROOT"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"
  exit 0
fi
export PYTHONPATH="$REPO_ROOT/13_base_method/grip-exp:$HNG/src:$HNG/scripts:${PYTHONPATH:-}"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
  "exec '$PYTHON' -u '$HNG/scripts/supervise_stage2_three_datasets.py' --root '$RUN_ROOT' --adopt-pid '$ADOPT_PID' >> '$RUN_ROOT/supervisor.log' 2>&1"
echo "Background Stage-2 queue: $SESSION"
echo "Status: $RUN_ROOT/STATUS.md"

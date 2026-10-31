#!/usr/bin/env bash
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
RUN_ROOT="$HNG/results/hpo/stage2_three_dataset_20261030"
SESSION=grip_stage2_watchdog
mkdir -p "$RUN_ROOT/logs"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
  "exec '$PYTHON' -u '$HNG/scripts/watch_stage2_queue.py' --root '$RUN_ROOT' >> '$RUN_ROOT/logs/watchdog.log' 2>&1"
echo "Background watchdog: $SESSION"

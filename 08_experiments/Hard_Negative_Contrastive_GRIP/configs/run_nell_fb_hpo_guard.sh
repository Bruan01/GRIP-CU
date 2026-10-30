#!/usr/bin/env bash
# Launch once in an independent tmux session; rerun to resume failed stages.
set -euo pipefail
HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
CAMPAIGN_ROOT="$HNG/results/hpo/nell_fb_guard_20261029"
SESSION=grip_nell_fb_hpo_guard
mkdir -p "$CAMPAIGN_ROOT"
if [[ -f "$CAMPAIGN_ROOT/CONTROLLER_HANDOFF.json" ]]; then
  exec bash "$HNG/configs/run_stage2_three_datasets.sh"
fi
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Monitor already has a tmux session: $SESSION"
  echo "Status: $CAMPAIGN_ROOT/STATUS.md"
  exit 0
fi
export PYTHONPATH="$REPO_ROOT/13_base_method/grip-exp:$HNG/src:$HNG/scripts:${PYTHONPATH:-}"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
  "exec '$PYTHON' -u '$HNG/scripts/supervise_nell_fb_hpo.py' --root '$CAMPAIGN_ROOT' >> '$CAMPAIGN_ROOT/supervisor.log' 2>&1"
echo "Started background monitor: $SESSION"
echo "Status: $CAMPAIGN_ROOT/STATUS.md"
echo "Log: $CAMPAIGN_ROOT/supervisor.log"

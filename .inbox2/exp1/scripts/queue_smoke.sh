#!/usr/bin/env bash
# Wait without using the GPU, then hand off to the isolated smoke launcher.
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_LOG="$EXP_DIR/runs/queue_20260928.log"
WAIT_PID="${WAIT_PID:-60981}"
mkdir -p "$EXP_DIR/runs"
exec >>"$QUEUE_LOG" 2>&1
echo "[$(date --iso-8601=seconds)] Queue armed; waiting for PID $WAIT_PID and NVIDIA device users to exit."
while true; do
  if kill -0 "$WAIT_PID" 2>/dev/null; then
    sleep 60
    continue
  fi
  if command -v fuser >/dev/null 2>&1 && fuser -s /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] GPU still in use; checking again in 60 seconds."
    sleep 60
    continue
  fi
  echo "[$(date --iso-8601=seconds)] Previous process ended and NVIDIA devices appear free; attempting launch."
  set +e
  RUN_ID=qwen05b_mixed_lora_smoke TMUX_SESSION=exp1-smoke-20260928 \
    bash "$EXP_DIR/scripts/launch_smoke.sh"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] Smoke launched successfully in tmux exp1-smoke-20260928."
    exit 0
  elif [[ $status -eq 3 ]]; then
    echo "[$(date --iso-8601=seconds)] GPU busy at handoff; retrying in 60 seconds."
    sleep 60
  else
    echo "[$(date --iso-8601=seconds)] Queue stopped: launcher returned status $status."
    exit "$status"
  fi
done

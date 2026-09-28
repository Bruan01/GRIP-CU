#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_DIR/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/.venv/bin/python}"
RUN_ID="${RUN_ID:-qwen05b_mixed_lora_smoke}"
TMUX_SESSION="${TMUX_SESSION:-exp1-smoke-$(date +%Y%m%d)}"
MODEL_DIR="${MODEL_DIR:-$EXP_DIR/models/Qwen2.5-0.5B-Instruct}"
MAX_STEPS="${MAX_STEPS:-8}"
RESUME="${RESUME:-0}"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists; attach with: tmux attach -t $TMUX_SESSION" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 2
fi
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing training")
print(f"GPU preflight: {torch.cuda.get_device_name(0)}; BF16={torch.cuda.is_bf16_supported()}")
PY
if command -v fuser >/dev/null 2>&1 && fuser -s /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null; then
  echo "NVIDIA devices are currently in use; refusing to compete for GPU." >&2
  fuser -v /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm 2>&1 || true
  exit 3
fi
if [[ ! -f "$EXP_DIR/inputs/nell23k_smoke_16_each.json" ]]; then
  echo "Missing isolated smoke input; run scripts/prepare_inputs.py first." >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing model snapshot; run scripts/fetch_model.py first." >&2
  exit 2
fi
RUN_DIR="$EXP_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
if [[ "$RESUME" == 1 ]]; then RESUME_ARG="--resume"; else RESUME_ARG=""; fi
COMMAND="cd '$EXP_DIR' && export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONHASHSEED=2026 && '$PYTHON' scripts/train_mixed_lora_smoke.py --model '$MODEL_DIR' --input '$EXP_DIR/inputs/nell23k_smoke_16_each.json' --run-id '$RUN_ID' --max-steps '$MAX_STEPS' $RESUME_ARG 2>&1 | tee -a '$RUN_DIR/run.log'; status=\${PIPESTATUS[0]}; echo \"exit_code=\$status\" | tee -a '$RUN_DIR/run.log'; exit \$status"
tmux new-session -d -s "$TMUX_SESSION" "bash -lc $(printf '%q' "$COMMAND")"
printf 'Started tmux session: %s\nAttach: tmux attach -t %s\nLog: %s\n' "$TMUX_SESSION" "$TMUX_SESSION" "$RUN_DIR/run.log"

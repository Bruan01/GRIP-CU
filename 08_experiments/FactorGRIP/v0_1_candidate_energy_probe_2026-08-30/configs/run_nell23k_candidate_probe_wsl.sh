#!/usr/bin/env bash
set -euo pipefail
VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
CONDA_ENV="${CONDA_ENV:-guardenv}"
RUN_ID="${RUN_ID:-wsl3090_nell23k_candidate_energy_20260830_01}"
INPUT_FILE="${INPUT_FILE:-$CODE_DIR/outputs/data/nell23k/recurrent_relation_prediction.json}"
RUN_ROOT="${RUN_ROOT:-$VERSION_DIR/results/runs}"
PARENT_RUN="${PARENT_RUN:-$VERSION_DIR/../../RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29/results/runs/wsl3090_nell23k_diag_cross_20260830_01_nell23k_diagnostic_cross}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$CODE_DIR/model_cache}"
MODEL_NAME="${MODEL_NAME:-qwen-0.5b}"
# Prefer an existing project cache, then the standard Hugging Face snapshot cache.
# Passing the resolved directory as --model_name lets the resolver use either
# layout without copying a ~1 GB checkpoint into this experiment directory.
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_CANDIDATES=("$MODEL_CACHE_DIR/Qwen--Qwen2.5-0.5B-Instruct")
  HF_SNAPSHOT_ROOT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
  if [[ -d "$HF_SNAPSHOT_ROOT" ]]; then
    # Use shell globbing instead of GNU find -printf/-mindepth so the wrapper
    # remains runnable from macOS BSD userland as well as WSL/Linux.
    for candidate in "$HF_SNAPSHOT_ROOT"/*; do
      [[ -d "$candidate" ]] || continue
      MODEL_CANDIDATES+=("$candidate")
    done
  fi
  for candidate in "${MODEL_CANDIDATES[@]}"; do
    if [[ -f "$candidate/config.json" && -f "$candidate/model.safetensors" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_NAME="$MODEL_PATH"
fi

command -v conda >/dev/null 2>&1 || { echo "error: conda is required" >&2; exit 1; }
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "missing input: $INPUT_FILE; run configs/prepare_nell23k.sh first" >&2; exit 1
fi
K1="$PARENT_RUN/train_k1/adapters/nell23k"
K2="$PARENT_RUN/train_k2/adapters/nell23k"
[[ -f "$K1/adapter_config.json" && -f "$K2/adapter_config.json" ]] || { echo "missing v1.1.1 adapters under $PARENT_RUN" >&2; exit 1; }

cd "$CODE_DIR"
RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
START_NS=$(date +%s%N)
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_candidate_energy_probe.py \
  --input_file "$INPUT_FILE" --run_root "$RUN_ROOT" --run_id "$RUN_ID" \
  --model_name "$MODEL_NAME" --model_source local --model_cache_dir "$MODEL_CACHE_DIR" \
  --local_files_only --adapter_k1 "$K1" --adapter_k2 "$K2" --device cuda \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$VERSION_DIR/logs/${RUN_ID}.log"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/analyze_candidate_energy.py \
  --input_file "$RUN_ROOT/$RUN_ID/predictions.jsonl" --output_dir "$RUN_ROOT/$RUN_ID/analysis" \
  2>&1 | tee -a "$VERSION_DIR/logs/${RUN_ID}.log"
END_NS=$(date +%s%N)
conda run --no-capture-output -n "$CONDA_ENV" python - "$RUN_ROOT/$RUN_ID/run.log" "$START_NS" "$END_NS" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); row=json.loads(p.read_text()) if p.exists() else {}
row["gpu_wall_time_seconds"]=(int(sys.argv[3])-int(sys.argv[2]))/1e9
p.write_text(json.dumps(row, indent=2)+"\n")
PY
echo "$RUN_ROOT/$RUN_ID" > "$VERSION_DIR/results/LAST_CANDIDATE_ENERGY_RUN.txt"
echo "candidate-energy probe completed: $RUN_ROOT/$RUN_ID"

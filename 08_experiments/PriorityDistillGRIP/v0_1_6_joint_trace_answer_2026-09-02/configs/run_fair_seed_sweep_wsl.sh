#!/usr/bin/env bash
set -Eeuo pipefail

# Fair direct-vs-joint seed sweep.  The existing seed-42 runs are retained;
# this script adds seeds 43 and 44 and then runs candidate-set diagnostics.
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/kieran/miniconda3/envs/guardenv/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}"
SEEDS="${SEEDS:-43 44}"
cd "$EXPERIMENT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results/logs

for seed in $SEEDS; do
  for protocol in direct joint; do
    if [[ "$protocol" == direct ]]; then
      config="configs/direct_answer_only_fair_20260903.json"
    else
      config="configs/graph_free_trace_joint_fair_20260903.json"
    fi
    run_id="wsl3090_v017_${protocol}_seed${seed}_20260903"
    out="results/runs/${run_id}"
    echo "[$(date -Is)] training ${run_id} (${CONDA_DEFAULT_ENV:-unknown})"
    "$PYTHON_EXECUTABLE" scripts/run_controlled.py \
      --config "$config" \
      --output-dir "$out" \
      --model-name-or-path "$MODEL_NAME_OR_PATH" \
      --training-seed "$seed" \
      --overwrite 2>&1 | tee "results/logs/${run_id}.log"
    selected="$($PYTHON_EXECUTABLE - "$out/selected_checkpoint_metrics.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])
PY
)"
    checkpoint="$out/checkpoints/$selected/adapter_model.pt"
    diag="$out/candidate_diagnostics_${selected}"
    "$PYTHON_EXECUTABLE" scripts/evaluate_constrained_answers.py \
      --config "$config" --checkpoint "$checkpoint" --output-dir "$diag/constrained" \
      --model-name-or-path "$MODEL_NAME_OR_PATH"
    "$PYTHON_EXECUTABLE" scripts/score_deployment_candidates.py \
      --config "$config" --checkpoint "$checkpoint" --output-dir "$diag/deployment" \
      --model-name-or-path "$MODEL_NAME_OR_PATH" --split test
  done
done

args=()
for seed in $SEEDS; do
  for protocol in direct joint; do
    args+=(--run-dir "results/runs/wsl3090_v017_${protocol}_seed${seed}_20260903")
  done
done
"$PYTHON_EXECUTABLE" scripts/aggregate_seed_results.py \
  "${args[@]}" --output results/fair_seed_sweep_20260903.json

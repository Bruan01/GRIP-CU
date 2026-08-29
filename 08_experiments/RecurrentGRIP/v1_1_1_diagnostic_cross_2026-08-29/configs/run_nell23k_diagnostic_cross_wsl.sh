#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
NELL_INPUT="${NELL_INPUT:-$CODE_DIR/outputs/data/nell23k/recurrent_relation_prediction.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$VERSION_DIR/results/runs/${RUN_ID}_nell23k_diagnostic_cross}"
CONTEXT_NODE_SAMPLES="${CONTEXT_NODE_SAMPLES:-32}"
CONTEXT_EDGE_SAMPLES="${CONTEXT_EDGE_SAMPLES:-224}"
CONTEXT_SAMPLING_SEED="${CONTEXT_SAMPLING_SEED:-2026}"
EXPECTED_CONTEXT_SHA256="${EXPECTED_CONTEXT_SHA256:-d4a9cf3f6d2deee090af5137a2b52f89f8ce335926eabbd344ece85729d72885}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "error: RUN_ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "error: run configs/setup_wsl3090.sh first" >&2
  exit 1
fi
if [[ ! -f "$NELL_INPUT" ]]; then
  echo "error: NELL23K recurrent input not found: $NELL_INPUT" >&2
  echo "run: bash configs/prepare_nell23k.sh" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "error: run directory already exists; choose a new RUN_ID: $RUN_DIR" >&2
  exit 1
fi
command -v timeout >/dev/null 2>&1 || { echo "error: GNU timeout is required in WSL2" >&2; exit 1; }

mkdir -p "$RUN_DIR"
cp "$VERSION_DIR/configs/nell23k_diagnostic_cross_qwen05b.json" "$RUN_DIR/config.json"
cp "$NELL_INPUT.stats.json" "$RUN_DIR/input_stats.json" 2>/dev/null || true
{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "dataset=NELL23K"
  echo "nell_input=$NELL_INPUT"
  date --iso-8601=seconds
  nvidia-smi
  "$PYTHON" "$VERSION_DIR/configs/verify_wsl3090.py"
  "$PYTHON" -m pip freeze 2>/dev/null || uv pip freeze --python "$PYTHON"
} > "$RUN_DIR/environment.txt" 2>&1

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

for TRAIN_K in 1 2; do
  SUBDIR="$RUN_DIR/train_k${TRAIN_K}"
  mkdir -p "$SUBDIR"
  timeout --signal=TERM --kill-after=5m 45m \
    "$PYTHON" scripts/run_recurrent_pilot.py \
      --input_file "$NELL_INPUT" \
      --output_file "$SUBDIR/predictions.jsonl" \
      --training_output_dir "$SUBDIR/trainer" \
      --model_name qwen-0.5b \
      --max_graphs 1 \
      --involve_qa_epochs 1 \
      --gen_max_length 24 \
      --wall_time_limit_minutes 30 \
      --hard_stop_minutes 45 \
      --seed 2026 \
      -- \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --learning_rate 2e-4 \
      --logging_steps 1 \
      --save_strategy no \
      --report_to none \
      --bf16 true \
      --recurrent_depth_train "$TRAIN_K" \
      --recurrent_depth_sweep 1 2 \
      --context_node_samples "$CONTEXT_NODE_SAMPLES" \
      --context_edge_samples "$CONTEXT_EDGE_SAMPLES" \
      --context_sampling_seed "$CONTEXT_SAMPLING_SEED" \
      --target_modules q_proj k_proj v_proj \
      --adapter_control all \
      --evaluation_device cuda \
      --require_cuda true \
      --adapter_output_dir "$SUBDIR/adapters" \
    2>&1 | tee "$SUBDIR/run.log"

  MANIFEST="$SUBDIR/adapters/nell23k/context_sampling_manifest.json"
  "$PYTHON" - "$MANIFEST" "$CONTEXT_NODE_SAMPLES" "$CONTEXT_EDGE_SAMPLES" "$EXPECTED_CONTEXT_SHA256" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_nodes = int(sys.argv[2])
expected_edges = int(sys.argv[3])
expected_sha256 = sys.argv[4]
manifest = json.loads(path.read_text(encoding="utf-8"))
assert manifest["sampling_strategy"] == (
    "node_random_train_qa_anchor_edge_relation_round_robin_v2"
), manifest
assert manifest["selected_node_count"] == expected_nodes, manifest
assert manifest["selected_edge_count"] == expected_edges, manifest
assert manifest["selected_relation_count"] == manifest["relation_count"] == 198, manifest
assert manifest["relation_coverage"] == 1.0, manifest
assert manifest["train_qa_count"] == 64, manifest
assert manifest["train_qa_fact_count"] == 64, manifest
assert manifest["train_qa_fact_covered_count"] == 64, manifest
assert manifest["train_qa_fact_coverage"] == 1.0, manifest
assert manifest["train_qa_relation_coverage"] == 1.0, manifest
assert manifest["train_qa_entity_endpoint_count"] == 128, manifest
assert manifest["train_qa_entity_covered_count"] == 128, manifest
assert manifest["train_qa_entity_coverage"] == 1.0, manifest
assert manifest["selection_sha256"] == expected_sha256, manifest
print(json.dumps({
    "context_manifest": str(path),
    "strategy": manifest["sampling_strategy"],
    "nodes": manifest["selected_node_count"],
    "edges": manifest["selected_edge_count"],
    "relations": manifest["selected_relation_count"],
    "relation_coverage": manifest["relation_coverage"],
    "train_qa_fact_coverage": manifest["train_qa_fact_coverage"],
    "train_qa_entity_coverage": manifest["train_qa_entity_coverage"],
    "selection_sha256": manifest["selection_sha256"],
}, ensure_ascii=False))
PY
done

cat "$RUN_DIR/train_k1/predictions.jsonl" "$RUN_DIR/train_k2/predictions.jsonl" \
  > "$RUN_DIR/predictions.jsonl"
"$PYTHON" recurrent_cross_audit.py \
  --run_dir "$RUN_DIR" \
  --output_file "$RUN_DIR/cross_run_audit.json" \
  2>&1 | tee "$RUN_DIR/cross_run_audit.log"
"$PYTHON" scripts/analyze_recurrent_results.py \
  --input_file "$RUN_DIR/predictions.jsonl" \
  --output_dir "$RUN_DIR/analysis" \
  2>&1 | tee "$RUN_DIR/analysis.log"

echo "$RUN_DIR" > "$VERSION_DIR/results/LAST_NELL23K_DIAGNOSTIC_CROSS_RUN.txt"
echo "NELL23K diagnostic cross completed: $RUN_DIR"

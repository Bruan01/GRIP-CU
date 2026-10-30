#!/usr/bin/env bash
# Resumable CoDEx-M SC-CL pipeline for the local RTX 3090.
# It waits for train-only task generation, trains the B1 adapter, then runs
# train-only scoring/bank construction, wiring, one-seed listed training, and
# evaluation. Each completed stage is reused when this supervisor is rerun.
set -Eeuo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HNG/../.." && pwd)"
BASELINE_DIR="${GRIP_BASELINE_DIR:-$REPO_ROOT/13_base_method/grip-exp}"
PYTHON="${PYTHON:-/home/kieran/miniconda3/envs/guardenv/bin/python}"
GENERIC_RUNNER="$HNG/configs/run_error_memory_wsl.sh"
TASK_CACHE="$REPO_ROOT/04_benchmark/data/codexm/tasks/cache_vllm_qwen25_7b_b8_gpu76_20261026"
TASK_STATE="$TASK_CACHE/run_state.json"
TASK_PLAN="$REPO_ROOT/04_benchmark/data/codexm/tasks/generation_plan.json"
TASK_FILE="$REPO_ROOT/04_benchmark/data/codexm/tasks/grip_codexm_tasks.json"
DATA_ROOT="$REPO_ROOT/04_benchmark/data/codexm"
RAW_DIR="$DATA_ROOT/raw"
EVAL_FILE="$DATA_ROOT/processed/processed_test.json"
RELATION_TEXT_FILE="$RAW_DIR/rel2text.json"
ENTITY_TEXT_FILE="$RAW_DIR/entity2text.txt"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/mnt/e/GRIP-CU/model_cache}"
SEED="${SEED:-2026}"

# Match CoDEx-M's GRIP recipe while using microbatch 2 + accumulation 256 to
# keep the effective batch at 512 with a safer 3090 memory peak.
B1_ROOT="$BASELINE_DIR/outputs/paper_runs/codexm_taskfile_3090"
S1_ADAPTER="$B1_ROOT/stage1_adapter"
B1_ADAPTER="$B1_ROOT/stage2_adapter"
RUN_ROOT="${RUN_ROOT:-$HNG/results/runs/codexm_sccl_3090_20261026}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$RUN_ROOT/artifacts}"
SCORES_DIR="$ARTIFACT_ROOT/b1_train_scores"
SAMPLER_DIR="$ARTIFACT_ROOT/structure_counterfactual_bank"
MANIFEST="$SAMPLER_DIR/counterfactual_bank.jsonl"
COUNTERFACTUAL_METADATA="$SAMPLER_DIR/metadata.json"
LISTED_ADAPTER="$RUN_ROOT/train/listed/adapter"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1
export PYTHONPATH="$BASELINE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export GRIP_BASELINE_DIR="$BASELINE_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

adapter_ready() {
  [[ -f "$1/adapter_config.json" && -f "$1/adapter_model.safetensors" ]]
}

wait_for_task_generation() {
  echo "[codexm-pipeline] waiting for train-only task generation: $TASK_STATE"
  while true; do
    local status="missing"
    local pid=""
    if [[ -f "$TASK_STATE" ]]; then
      read -r status pid < <("$PYTHON" - "$TASK_STATE" <<'PY'
import json, sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(state.get("status", "unknown"), state.get("pid", ""))
PY
)
    fi
    case "$status" in
      complete)
        [[ -f "$TASK_FILE" ]] || { echo "[codexm-pipeline] completed state has no task file" >&2; exit 1; }
        echo "[codexm-pipeline] task generation completed"
        return
        ;;
      failed|failed_resumable|interrupted)
        echo "[codexm-pipeline] task generation stopped with status=$status; resume it before restarting this pipeline" >&2
        exit 1
        ;;
      running)
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
          echo "[codexm-pipeline] task generation state is running but PID $pid is absent" >&2
          exit 1
        fi
        echo "[codexm-pipeline] task generation still running; next check in 60s"
        sleep 60
        ;;
      *)
        echo "[codexm-pipeline] no completed run state yet; next check in 60s"
        sleep 60
        ;;
    esac
  done
}

record_and_validate_task_file() {
  "$PYTHON" - "$REPO_ROOT" "$TASK_FILE" "$TASK_STATE" "$TASK_PLAN" <<'PY'
import json, sys
from pathlib import Path

root, task_path, state_path, plan_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "04_benchmark/preprocessing"))
sys.path.insert(0, str(root / "08_experiments/Hard_Negative_Contrastive_GRIP/src"))
from prepare_benchmark_data import atomic_json_dump, file_record
from hard_negative_grip.official_lists import load_train_relation_order

task = json.loads(task_path.read_text(encoding="utf-8"))
state = json.loads(state_path.read_text(encoding="utf-8"))
if state.get("status") != "complete" or task.get("generation_status") != "complete":
    raise SystemExit("CoDEx-M task generation is not complete")
config = task.get("generation_config") or {}
expected_input = (root / "04_benchmark/data/codexm/processed/processed_train_graph.json").resolve()
expected_reference = (root / "04_benchmark/data/codexm/processed/processed_train_reference.json").resolve()
for key, expected in (("input_file", expected_input), ("reference_file", expected_reference)):
    actual = Path(config.get(key, ""))
    if not actual.is_absolute():
        actual = (root / actual).resolve()
    if actual != expected:
        raise SystemExit(f"train-only gate failed: {key}={actual}, expected {expected}")
if config.get("dataset") != "codexm" or config.get("input_split") != "train" or config.get("reference_split") != "train":
    raise SystemExit("task generation config must use CoDEx-M train graph and train references")
counts = task.get("counts") or {}
if int(counts.get("graphs", 0)) != 1:
    raise SystemExit(f"expected one CoDEx-M training graph, got {counts.get('graphs')}")
if int(counts.get("context_samples", 0)) <= 0 or int(counts.get("qa_samples", 0)) <= 3253:
    raise SystemExit(f"full task budget missing: {counts}")
relation_order = load_train_relation_order(
    root / "04_benchmark/data/codexm/raw",
    relation_text_file=root / "04_benchmark/data/codexm/raw/rel2text.json",
)
if len(relation_order) != 51:
    raise SystemExit(f"expected 51 train relations, got {len(relation_order)}")

plan = json.loads(plan_path.read_text(encoding="utf-8"))
plan["status"] = "complete"
plan["completed_at_utc"] = state.get("completed_at_utc")
plan["counts"] = counts
atomic_json_dump(plan_path, plan)
manifest_path = root / "04_benchmark/data/codexm/manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
generated = manifest.setdefault("generated_files", {})
generated["tasks/grip_codexm_tasks.json"] = file_record(task_path)
generated["tasks/generation_plan.json"] = file_record(plan_path)
atomic_json_dump(manifest_path, manifest)
print(json.dumps({
    "status": "validated",
    "task_file": str(task_path),
    "counts": counts,
    "train_relations": len(relation_order),
    "reference_split": config.get("reference_split"),
}, ensure_ascii=False), flush=True)
PY
}

run_stage() {
  local step="$1"
  local run_dir="$2"
  shift 2
  echo "[codexm-pipeline] step=$step start=$(date --iso-8601=seconds)"
  env STEP="$step" DATASET=codexm NEGATIVE_SOURCE=structure_counterfactual \
    RUN_ID="codexm_sccl_3090_20261026_${step}" \
    RUN_DIR="$run_dir" RUN_VARIANT=codexm_seed2026 \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" SCORES_DIR="$SCORES_DIR" \
    SAMPLER_DIR="$SAMPLER_DIR" MANIFEST="$MANIFEST" \
    COUNTERFACTUAL_METADATA="$COUNTERFACTUAL_METADATA" \
    INPUT_FILE="$TASK_FILE" EVAL_FILE="$EVAL_FILE" RAW_DIR="$RAW_DIR" \
    RELATION_TEXT_FILE="$RELATION_TEXT_FILE" ENTITY_TEXT_FILE="$ENTITY_TEXT_FILE" \
    S1_ADAPTER="$S1_ADAPTER" B1_ADAPTER="$B1_ADAPTER" \
    MODEL_CACHE_DIR="$MODEL_CACHE_DIR" SEED="$SEED" \
    EXPECTED_LISTED=-1 LISTED_NEGATIVE_K=9 \
    LAMBDA_CANDIDATE=1.0 CONTRASTIVE_TEMPERATURE=1.0 \
    TRAIN_LEARNING_RATE=1e-3 TRAIN_WEIGHT_DECAY=1e-4 TRAIN_MAX_GRAD_NORM=1.0 \
    TRAIN_LORA_R=4 TRAIN_LORA_ALPHA=8 INVOLVE_QA_EPOCHS=5 \
    PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=256 \
    CANDIDATE_BATCH_SIZE=8 PROGRESS_EVERY=25 SKIP_TMUX=1 \
    "$@" bash "$GENERIC_RUNNER"
  echo "[codexm-pipeline] step=$step finished=$(date --iso-8601=seconds)"
}

wait_for_task_generation
record_and_validate_task_file

run_stage preflight "$RUN_ROOT/preflight"

if ! adapter_ready "$S1_ADAPTER" || ! adapter_ready "$B1_ADAPTER"; then
  mkdir -p "$B1_ROOT"
  echo "[codexm-pipeline] training CoDEx-M B1 adapter start=$(date --iso-8601=seconds)"
  cd "$BASELINE_DIR"
  "$PYTHON" scripts/train_task_file_adapter.py \
    --task_file "$TASK_FILE" \
    --output_dir "$B1_ROOT" \
    --model_name qwen-7b \
    --model_source local \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --local_files_only \
    --dtype bfloat16 \
    --tokenize_max_length 4096 \
    --stage1_epochs 1 \
    --stage2_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 256 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --adam_epsilon 1e-8 \
    --lora_r 4 \
    --lora_alpha 8 \
    --target_modules down_proj up_proj gate_proj \
    --s1_stop_loss_threshold 0.15 \
    --s2_stop_loss_threshold 0.15 \
    --s1_min_epoch 1 \
    --s2_min_epoch 1 \
    --gradient_checkpointing \
    --stage1_checkpoint_interval 25 \
    --stage2_checkpoint_interval 10 \
    --seed "$SEED" 2>&1 | tee -a "$B1_ROOT/launcher.log"
else
  echo "[codexm-pipeline] reusing complete CoDEx-M B1 adapters"
fi
adapter_ready "$S1_ADAPTER" || { echo "missing CoDEx-M Stage-1 adapter: $S1_ADAPTER" >&2; exit 1; }
adapter_ready "$B1_ADAPTER" || { echo "missing CoDEx-M B1 adapter: $B1_ADAPTER" >&2; exit 1; }

if [[ ! -f "$SCORES_DIR/run_stats.json" || ! -f "$SCORES_DIR/candidate_scores.jsonl" ]]; then
  run_stage score "$RUN_ROOT/score"
else
  echo "[codexm-pipeline] reusing complete B1 train-score artifact"
fi

if [[ ! -f "$MANIFEST" || ! -f "$COUNTERFACTUAL_METADATA" || \
      ! -f "$SAMPLER_DIR/sampler_audit.jsonl" || ! -f "$SAMPLER_DIR/relation_profiles.json" ]]; then
  run_stage freeze "$RUN_ROOT/freeze"
else
  echo "[codexm-pipeline] reusing complete frozen CoDEx-M bank"
fi

if [[ ! -f "$RUN_ROOT/wire/wiring.json" ]]; then
  run_stage wire "$RUN_ROOT/wire"
else
  echo "[codexm-pipeline] reusing completed wiring gate"
fi

if ! adapter_ready "$LISTED_ADAPTER"; then
  run_stage train "$RUN_ROOT/train" \
    ALLOW_TRAIN=1 DEFER_EVAL=1 SAVE_STEPS=10 SAVE_TOTAL_LIMIT=2
else
  echo "[codexm-pipeline] reusing completed CoDEx-M SC-CL adapter"
fi

for pair in \
  "b1:$B1_ADAPTER:$RUN_ROOT/eval/b1" \
  "sccl:$LISTED_ADAPTER:$RUN_ROOT/eval/sccl"; do
  IFS=: read -r label adapter output_dir <<<"$pair"
  if [[ ! -f "$output_dir/summary.json" ]]; then
    mkdir -p "$output_dir"
    cd "$BASELINE_DIR"
    "$PYTHON" "$HNG/scripts/evaluate_adapter.py" \
      --adapter "$adapter" \
      --eval_file "$EVAL_FILE" \
      --output_dir "$output_dir" \
      --model_name qwen-7b \
      --model_cache_dir "$MODEL_CACHE_DIR" \
      --dtype bfloat16 \
      --gen_max_length 1000 \
      --progress_every 25 \
      --resume \
      --label "$label"
  else
    echo "[codexm-pipeline] reusing evaluation: $output_dir/summary.json"
  fi
done

echo "[codexm-pipeline] complete=$(date --iso-8601=seconds)"

#!/usr/bin/env bash
# Listed-contrastive on the frozen Random-K manifest, full paper task file.
#
# Protocol after Prompt 7: CPU-wire the frozen IDs onto grip_nell23k_tasks.json,
# then train Random-K only. Same 20260913 Stage-1 adapter, same frozen B1,
# same 7B recipe (batch=1, accum=512, 10 epochs). That is the experiment-H
# budget (~230 listed updates), not the retired 64-QA / 10-step smoke.
# Eval stays on the aligned 32/64 smoke split so this run is comparable to H.
#
# Do not rescore 3253x198. Top-K Hard / Coverage-Adaptive K stay frozen until
# Random-K has a real budget.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"
OLD_RUN="${OLD_RUN:-$HNG/results/runs/20260913_qwen7b_tasks_listed_vs_b1_qwen-7b_smoke}"
S1_ADAPTER="${S1_ADAPTER:-$OLD_RUN/s1_adapter}"
INPUT_FILE="${INPUT_FILE:-$HNG/grip_nell23k_tasks.json}"
EVAL_FILE="${EVAL_FILE:-$HNG/data/nell23k/recurrent_relation_prediction.aligned.json}"
RAW_DIR="${RAW_DIR:-$CODE_DIR/data/raw_datasets/nell23k}"
SAMPLER_DIR="${SAMPLER_DIR:-$HNG/results/runs/20260926_shared_pool_samplers}"
VARIANT="${VARIANT:-random_k}"
SEED="${SEED:-2026}"
EXPECTED_LISTED="${EXPECTED_LISTED:-3253}"
LAST_RUN_FILE="$HNG/results/LAST_SHARED_POOL_SAMPLER_TRAIN_RUN.txt"

if [[ "$VARIANT" != "random_k" && "${ALLOW_NON_RANDOM:-0}" != "1" ]]; then
  echo "error: default protocol trains Random-K only; set ALLOW_NON_RANDOM=1 to override" >&2
  echo "  got VARIANT=$VARIANT" >&2
  exit 1
fi
case "$VARIANT" in
  random_k|top_k_hard|coverage_adaptive_k) ;;
  *)
    echo "error: VARIANT must be random_k, top_k_hard, or coverage_adaptive_k" >&2
    exit 1
    ;;
esac

RUN_ID="${RUN_ID:-20260926_shared_pool_${VARIANT}_full}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}}"
if [[ "${RESUME_LAST:-0}" == "1" ]]; then
  if [[ ! -f "$LAST_RUN_FILE" ]]; then
    echo "error: RESUME_LAST=1 but missing $LAST_RUN_FILE" >&2
    exit 1
  fi
  RUN_DIR="$(tr -d '\n' < "$LAST_RUN_FILE")"
fi
TMUX_SESSION="${TMUX_SESSION:-shared-pool-${VARIANT}-full-20260926}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: Python environment not found: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'; then
  echo "error: PyTorch cannot see a CUDA device; refusing to run CPU experiment" >&2
  exit 1
fi
MODEL_CACHE="$CODE_DIR/model_cache"
MODEL_DIR="$MODEL_CACHE/Qwen--Qwen2.5-7B-Instruct"
if ! PYTHONPATH="$CODE_DIR" "$PYTHON" -c "
from pathlib import Path
from models.utils import _is_complete_model_dir
raise SystemExit(0 if _is_complete_model_dir(Path(r'$MODEL_DIR')) else 1)
"; then
  echo "error: Qwen2.5-7B cache missing or incomplete: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -f "$S1_ADAPTER/adapter_config.json" || ! -f "$S1_ADAPTER/adapter_model.safetensors" ]]; then
  echo "error: incomplete Stage-1 adapter: $S1_ADAPTER" >&2
  exit 1
fi
for required in "$INPUT_FILE" "$EVAL_FILE" "$RAW_DIR/train.txt" "$SAMPLER_DIR/policy.json" "$SAMPLER_DIR/${VARIANT}.jsonl"; do
  if [[ ! -e "$required" ]]; then
    echo "error: missing required path: $required" >&2
    exit 1
  fi
done
if [[ "${FORCE_NEW:-0}" == "1" && -e "$RUN_DIR" ]]; then
  echo "error: FORCE_NEW=1 but run directory already exists: $RUN_DIR" >&2
  exit 1
fi

# shellcheck source=tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$LAST_RUN_FILE"
WIRE_DIR="${WIRE_DIR:-$SAMPLER_DIR/wired_full}"
MANIFEST="$SAMPLER_DIR/${VARIANT}.jsonl"

export PYTHONPATH="$CODE_DIR:$HNG/src:$HNG/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$CODE_DIR"

log() {
  echo "$*" | tee -a "$RUN_DIR/run.log"
}

{
  echo "variant=$VARIANT"
  echo "run_dir=$RUN_DIR"
  echo "s1_adapter=$S1_ADAPTER"
  echo "input_file=$INPUT_FILE"
  echo "eval_file=$EVAL_FILE"
  echo "sampler_dir=$SAMPLER_DIR"
  echo "expected_listed=$EXPECTED_LISTED"
  echo "seed=$SEED"
  echo "tmux_session=$TMUX_SESSION"
  echo "listed_negative_source=score_hard"
  echo "note=full paper-task Random-K listed train on frozen train-only manifests; not 64-QA / 10-step"
  date --iso-8601=seconds
  "$PYTHON" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  echo -n "manifest_${VARIANT}_sha256="
  sha256sum "$MANIFEST" | awk '{print $1}'
} >> "$RUN_DIR/environment.txt" 2>&1

if [[ ! -f "$WIRE_DIR/wiring.json" ]]; then
  log "[wire] missing $WIRE_DIR/wiring.json; running CPU gate"
  OUTPUT_DIR="$WIRE_DIR" SAMPLER_DIR="$SAMPLER_DIR" INPUT_FILE="$INPUT_FILE" \
    RAW_DIR="$RAW_DIR" EXPECTED_LISTED="$EXPECTED_LISTED" SKIP_TMUX=1 \
    bash "$HNG/configs/run_wire_shared_pool_samplers.sh" | tee -a "$RUN_DIR/run.log"
fi

log "[budget] refuse 64-QA / 10-step; require full paper task file and accum=512"
"$PYTHON" - <<PY
import json
import sys
from pathlib import Path

sys.path.insert(0, "$HNG/src")
from hard_negative_grip.shared_pool_samplers import refuse_underfit_listed_budget
from hard_negative_grip.task_file import is_grip_task_file, load_json_payload

payload = load_json_payload(Path("$INPUT_FILE"))
if not is_grip_task_file(payload):
    raise SystemExit("$INPUT_FILE is not a GRIP task file")
n_qa = len(payload["qa_samples"])
if n_qa <= int("$EXPECTED_LISTED"):
    raise SystemExit(
        f"task file has {n_qa} QA; full listed training needs the paper task file "
        f"(> {int('$EXPECTED_LISTED')} rows), not a matchable-only slice"
    )
budget = refuse_underfit_listed_budget(n_qa)
Path("$RUN_DIR/budget.json").write_text(
    json.dumps(budget, ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
print(json.dumps(budget, ensure_ascii=False), flush=True)
PY

log "[wire] attach frozen $VARIANT to $INPUT_FILE"
"$PYTHON" "$HNG/scripts/verify_shared_pool_sampler_wiring.py" \
  --task_file "$INPUT_FILE" \
  --sampler_dir "$SAMPLER_DIR" \
  --raw_dir "$RAW_DIR" \
  --expected_listed "$EXPECTED_LISTED" \
  --variants "$VARIANT" \
  --output "$RUN_DIR/wiring.json" 2>&1 | tee -a "$RUN_DIR/run.log"

TRAIN_FLAGS=()
if [[ "${NO_RESUME:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--no_resume)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_FLAGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
skip_flags=()
if [[ "${SKIP_TRAIN:-0}" == "1" && -f "$RUN_DIR/listed/adapter/adapter_config.json" ]]; then
  skip_flags+=(--skip_train)
fi

LISTED_RUN="$RUN_DIR" B1_RUN="$OLD_RUN" bash "$HNG/configs/attach_frozen_b1.sh" | tee -a "$RUN_DIR/run.log"
log "[launch] variant=$VARIANT manifest=$MANIFEST listed=$EXPECTED_LISTED accum=512"
set +e
"$PYTHON" "$HNG/scripts/train_listed_contrastive.py" \
  --input_file "$INPUT_FILE" \
  --eval_file "$EVAL_FILE" \
  --output_dir "$RUN_DIR" \
  --stage listed \
  --s1_adapter "$S1_ADAPTER" \
  --model_name qwen-7b \
  --model_cache_dir "$CODE_DIR/model_cache" \
  --lora_r 4 \
  --lora_alpha 8 \
  --target_modules down_proj up_proj gate_proj \
  --num_train_epochs 1 \
  --involve_qa_epochs 10 \
  --s1_stop_loss_threshold 0.15 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 512 \
  --s1_gradient_checkpointing \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --max_grad_norm 1.0 \
  --lambda_candidate 1.0 \
  --temperature 1.0 \
  --gen_max_length 32 \
  --seed "$SEED" \
  --listed_negative_source score_hard \
  --score_hard_manifest "$MANIFEST" \
  --raw_dir "$RAW_DIR" \
  --save_steps "${SAVE_STEPS:-10}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  "${TRAIN_FLAGS[@]}" \
  "${skip_flags[@]}" \
  2>&1 | tee -a "$RUN_DIR/run.log"
rc=${PIPESTATUS[0]}
set -e
log "[launch] variant=$VARIANT finished exit=$rc $(date --iso-8601=seconds)"
if [[ "$rc" -eq 0 ]]; then
  LISTED_RUN="$RUN_DIR" B1_RUN="$OLD_RUN" INPUT_FILE="$INPUT_FILE" S1_ADAPTER="$S1_ADAPTER" \
    bash "$HNG/configs/compare_listed_to_frozen_b1.sh" | tee -a "$RUN_DIR/run.log"
fi
exit "$rc"

#!/usr/bin/env bash
# DPO 冒烟测试 —— 验证训练流程走通。
#
# 冒烟（用 qwen-0.5b，不要 S2 adapter，只走 5 步）:
#   SKIP_TMUX=1 bash .inbox/run_dpo_smoke.sh
#
# 在 7B adapter 上继续 DPO 训练:
#   TMUX_SESSION=dpo-7b-20260918 \
#     MODEL_NAME=qwen-7b \
#     S2_ADAPTER=/path/to/listed/adapter \
#     SCALE=full \
#     bash .inbox/run_dpo_smoke.sh
#
# 断点续传:
#   RESUME_LAST=1 bash .inbox/run_dpo_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."
HNG="$PWD/08_experiments/Hard_Negative_Contrastive_GRIP"
VERSION_DIR="$(cd "$HNG/../RecurrentGRIP/v1_1_nell23k_first_2026-08-29" && pwd)"
CODE_DIR="$VERSION_DIR/grip-exp"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if python3 -c "import trl" 2>/dev/null; then
    PYTHON="$(command -v python3)"
  else
    PYTHON="$CODE_DIR/.venv/bin/python"
  fi
fi

# === 路径配置 ===
TASK_JSON="${TASK_JSON:-$HNG/grip_nell23k_tasks.json}"
# S2_ADAPTER 为空则跳过，直接用基座+新LoRA
S2_ADAPTER="${S2_ADAPTER:-}"
MODEL_NAME="${MODEL_NAME:-qwen-0.5b}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$CODE_DIR/model_cache}"

# === 运行目录 ===
SCALE="${SCALE:-smoke}"   # smoke=5步, full=全量
RUN_ID="${RUN_ID:-dpo-${SCALE}-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$HNG/results/runs/${RUN_ID}}"
LAST_RUN_FILE="$HNG/results/LAST_DPO_RUN.txt"
if [[ "${RESUME_LAST:-0}" == "1" ]]; then
  if [[ ! -f "$LAST_RUN_FILE" ]]; then
    echo "error: RESUME_LAST=1 but missing $LAST_RUN_FILE" >&2
    exit 1
  fi
  RUN_DIR="$(tr -d '\n' < "$LAST_RUN_FILE")"
fi
TMUX_SESSION="${TMUX_SESSION:-dpo-${SCALE}-${RUN_ID}}"

# === 数据文件 ===
DPO_DATA="${DPO_DATA:-$HNG/.inbox/dpo_${SCALE}.jsonl}"

echo "=== DPO Config ==="
echo "  SCALE=$SCALE"
echo "  MODEL=$MODEL_NAME"
echo "  S2_ADAPTER=${S2_ADAPTER:-"(none, fresh LoRA)"}"
echo "  DPO_DATA=$DPO_DATA"
echo "  RUN_DIR=$RUN_DIR"
echo "  TMUX_SESSION=$TMUX_SESSION"
echo ""

# 检查依赖
if ! "$PYTHON" -c "import trl" 2>/dev/null; then
  echo "error: trl 未安装。请先: pip install trl"
  exit 1
fi
if [[ -n "$S2_ADAPTER" && ! -d "$S2_ADAPTER" ]]; then
  echo "error: Stage-2 adapter 不存在: $S2_ADAPTER"
  exit 1
fi
if [[ ! -f "$TASK_JSON" ]]; then
  echo "error: task json 不存在: $TASK_JSON"
  exit 1
fi

# tmux 守卫
# shellcheck source=configs/tmux_guard.sh
source "$HNG/configs/tmux_guard.sh"
tmux_guard_reexec "$0" "$@"

mkdir -p "$RUN_DIR"

# === Step 1: 构建 DPO 数据集 ===
BUILD_SCRIPT="$PWD/.inbox/build_dpo_data.py"
BUILD_FLAGS=(
  --task_json "$TASK_JSON"
  --output "$DPO_DATA"
)
if [[ "$SCALE" == "smoke" ]]; then
  BUILD_FLAGS+=(--max_samples 30)
fi
echo "[step 1] building DPO dataset..." | tee -a "$RUN_DIR/run.log"
"$PYTHON" "$BUILD_SCRIPT" "${BUILD_FLAGS[@]}" 2>&1 | tee -a "$RUN_DIR/run.log"
echo "" >> "$RUN_DIR/run.log"

# === Step 2: DPO 训练 ===
TRAIN_SCRIPT="$PWD/.inbox/train_dpo_smoke.py"
TRAIN_FLAGS=(
  --dpo_data "$DPO_DATA"
  --output_dir "$RUN_DIR"
  --model_name "$MODEL_NAME"
  --model_cache_dir "$MODEL_CACHE_DIR"
  --lora_r 4
  --lora_alpha 8
  --target_modules down_proj up_proj gate_proj
  --learning_rate 1e-5
  --beta 0.1
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 4
  --save_steps 10
  --save_total_limit 2
)
if [[ -n "$S2_ADAPTER" ]]; then
  TRAIN_FLAGS+=(--s2_adapter "$S2_ADAPTER")
fi
if [[ "$SCALE" == "smoke" ]]; then
  TRAIN_FLAGS+=(--max_steps 5)
else
  TRAIN_FLAGS+=(--max_steps 0 --num_train_epochs 3)
fi
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--skip_train)
fi

echo "[step 2] DPO training (SCALE=$SCALE)..." | tee -a "$RUN_DIR/run.log"
export PYTHONPATH="$CODE_DIR:$HNG/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set +e
"$PYTHON" "$TRAIN_SCRIPT" "${TRAIN_FLAGS[@]}" 2>&1 | tee -a "$RUN_DIR/run.log"
rc=${PIPESTATUS[0]}
set -e
echo "[step 2] DPO training finished exit=$rc" | tee -a "$RUN_DIR/run.log"
echo "$RUN_DIR" > "$LAST_RUN_FILE"

if [[ "$rc" -eq 0 ]]; then
  echo "[done] DPO smoke test passed!" | tee -a "$RUN_DIR/run.log"
  echo "  adapter: $RUN_DIR/dpo_adapter"
  echo "  log:     $RUN_DIR/run.log"
  if [[ -z "${TMUX:-}" ]]; then
    echo "  tmux:    tmux attach -t $TMUX_SESSION"
  fi
else
  echo "[fail] DPO smoke test failed (exit=$rc). check $RUN_DIR/run.log" | tee -a "$RUN_DIR/run.log"
fi
exit "$rc"
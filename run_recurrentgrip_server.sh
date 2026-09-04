#!/usr/bin/env bash
# =============================================================================
# RecurrentGRIP NELL23K 涨点实验 —— 共享服务器一键脚本
#
# 用法（scp 整个项目到服务器后，在项目根目录执行）：
#   bash run_recurrentgrip_server.sh              # 默认跑全流程：env→data→smoke→pilot→analyze
#   bash run_recurrentgrip_server.sh env          # 只装环境
#   bash run_recurrentgrip_server.sh data         # 只生成 NELL23K 数据
#   bash run_recurrentgrip_server.sh smoke        # 只跑 smoke（0.5B, K=1/2）
#   bash run_recurrentgrip_server.sh pilot        # 只跑 pilot（0.5B, K=1/2/3/4）
#   bash run_recurrentgrip_server.sh analyze      # 只分析最新一次 run
#   bash run_recurrentgrip_server.sh baseline     # 复现 baseline GRIP（7B，慢，可选）
#
# 可选环境变量：
#   GPU_INDEX=0/1            指定用哪张卡；不设则自动选空闲显存最大的卡
#   PYTORCH_CUDA_INDEX=URL   服务器 CUDA 版本特殊时，指定 torch 下载源
#   HF_ENDPOINT=https://hf-mirror.com   国内服务器访问 HF 的镜像
#
# 安全设计（针对共享服务器，不影响其他用户）：
#   - 所有文件都写在本项目目录内，不碰系统路径、不碰其他用户 home
#   - 用 CUDA_VISIBLE_DEVICES 固定单卡，避免抢别人的 GPU
#   - HF/ModelScope/torch 缓存全部指向项目内 .cache，不污染共享缓存
#   - 训练用 timeout 兜底，避免失控进程
#   - 不 sudo、不改系统环境变量、不改全局 pip
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V11="$ROOT/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29"
BASE="$ROOT/13_base_method/grip-exp/grip-exp"
LOG="$ROOT/server_run.log"

# ---- 缓存隔离：全部指向项目内，不污染共享缓存 ----
export HF_HOME="$ROOT/.cache/huggingface"
export MODELSCOPE_CACHE="$ROOT/.cache/modelscope"
export TORCH_HOME="$ROOT/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1
# uv 常装在 ~/.local/bin，非交互 shell 不在 PATH 里，手动补上
export PATH="$HOME/.local/bin:$PATH"
# 国内服务器访问不了 huggingface.co，走 hf-mirror 镜像（服务器已确认可达）
export HF_ENDPOINT="https://hf-mirror.com"
# 国内服务器 uv 下载 python 慢时，取消下一行注释（使用清华镜像）：
# export UV_PYTHON_INSTALL_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/python-build-standalone"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "ERROR: $*"; exit 1; }

# ---- 选卡：默认自动选空闲显存最大的卡 ----
select_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi 不可用，确认服务器有 NVIDIA 驱动"
  if [[ -z "${GPU_INDEX:-}" ]]; then
    GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')"
    [[ -n "$GPU_INDEX" ]] || die "无法从 nvidia-smi 读到 GPU"
  fi
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  log "使用 GPU: cuda:$GPU_INDEX"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/    /'
}

# ---- 环境（recurrent 主环境） ----
setup_env() {
  select_gpu
  cd "$V11/grip-exp"
  command -v uv >/dev/null 2>&1 || die "需要 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"
  if [[ ! -x .venv/bin/python ]]; then
    log "创建 venv + 安装依赖（torch 2.7.1 / transformers 4.56.1 / peft 0.17.1 ...）"
    # 优先复用服务器已有 python(>=3.11)，避免从 github 下载；否则让 uv 下载 3.11
    local base_py=""
    for cand in "$(command -v python3 2>/dev/null)" /opt/anaconda3/bin/python3 /usr/bin/python3; do
      [[ -n "$cand" && -x "$cand" ]] || continue
      if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        base_py="$cand"; break
      fi
    done
    if [[ -n "$base_py" ]]; then
      log "复用已有 python: $base_py"
      uv venv --python "$base_py" .venv
    else
      log "未找到 python>=3.11，让 uv 下载 3.11"
      uv venv --python 3.11 .venv
    fi
    if [[ -n "${PYTORCH_CUDA_INDEX:-}" ]]; then
      uv pip install --python .venv/bin/python "torch==2.7.1" --index-url "$PYTORCH_CUDA_INDEX"
    else
      uv pip install --python .venv/bin/python "torch==2.7.1"
    fi
    uv pip install --python .venv/bin/python \
      "transformers==4.56.1" "peft==0.17.1" "accelerate==1.10.1" "datasets==4.0.0" \
      "numpy==2.2.6" "scipy==1.16.1" "torch-geometric==2.6.1" "openai==1.107.0" \
      "anthropic==0.66.0" "tiktoken==0.11.0" "tenacity==9.1.2" "pytest>=8,<9" tqdm
  fi
  log "验证 CUDA / 3090 / BF16 ..."
  .venv/bin/python "$V11/configs/verify_wsl3090.py" || die "GPU 验证失败（不是 3090 或 CUDA 不可用）"
  log "环境就绪"
}

# ---- 数据（纯 Python，确定性；已存在则跳过） ----
prepare_data() {
  cd "$V11"
  local PY="grip-exp/.venv/bin/python"
  if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || command -v python || true)"
    [[ -n "$PY" ]] || die "找不到 python3（可先跑 env 阶段建立 venv）"
  fi
  local smoke_out="grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json"
  local pilot_out="grip-exp/outputs/data/nell23k/recurrent_relation_prediction_pilot.json"
  if [[ ! -f "$smoke_out" ]]; then
    log "生成 smoke 数据 (64/32/64)"
    MAX_TRAIN_QUESTIONS=64 MAX_VALIDATION_QUESTIONS=32 MAX_TEST_QUESTIONS=64 SEED=2026 \
      PYTHON="$PY" bash configs/prepare_nell23k.sh
  else
    log "smoke 数据已存在，跳过"
  fi
  if [[ ! -f "$pilot_out" ]]; then
    log "生成 pilot 数据 (512/128/512)"
    MAX_TRAIN_QUESTIONS=512 MAX_VALIDATION_QUESTIONS=128 MAX_TEST_QUESTIONS=512 SEED=2026 \
      NELL_OUTPUT="$V11/$pilot_out" PYTHON="$PY" bash configs/prepare_nell23k.sh
  else
    log "pilot 数据已存在，跳过"
  fi
}

# ---- smoke ----
run_smoke() {
  select_gpu
  cd "$V11"
  local RUN_ID="${RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}"
  log "运行 smoke（0.5B, K=1/2）RUN_ID=$RUN_ID"
  RUN_ID="$RUN_ID" PYTHON="$V11/grip-exp/.venv/bin/python" bash configs/run_nell23k_smoke_wsl.sh
}

# ---- pilot ----
run_pilot() {
  select_gpu
  cd "$V11"
  local RUN_ID="${RUN_ID:-pilot_$(date +%Y%m%d_%H%M%S)}"
  log "运行 pilot（0.5B, K=1/2/3/4）RUN_ID=$RUN_ID"
  RUN_ID="$RUN_ID" PYTHON="$V11/grip-exp/.venv/bin/python" bash configs/run_nell23k_pilot_wsl.sh
}

# ---- 分析 ----
run_analyze() {
  cd "$V11"
  local RUN_DIR="${RUN_DIR:-}"
  if [[ -z "$RUN_DIR" ]]; then
    if [[ -f results/LAST_NELL23K_PILOT_RUN.txt ]]; then
      RUN_DIR="$(cat results/LAST_NELL23K_PILOT_RUN.txt)"
    elif [[ -f results/LAST_NELL23K_SMOKE_RUN.txt ]]; then
      RUN_DIR="$(cat results/LAST_NELL23K_SMOKE_RUN.txt)"
    else
      die "找不到 run 记录，先跑 smoke 或 pilot"
    fi
  fi
  log "分析 run: $RUN_DIR"
  RUN_DIR="$RUN_DIR" PYTHON="$V11/grip-exp/.venv/bin/python" bash configs/analyze_nell23k.sh
}

# ---- baseline GRIP（7B，可选，慢） ----
run_baseline() {
  select_gpu
  cd "$BASE"
  command -v uv >/dev/null 2>&1 || die "需要 uv"
  if [[ ! -x .venv/bin/python ]]; then
    log "创建 baseline venv"
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python "torch==2.7.1"
    uv pip install --python .venv/bin/python modelscope transformers peft accelerate datasets
  fi
  log "下载 Qwen2.5-7B（ModelScope，约 15GB，一次性）"
  .venv/bin/python scripts/download_model.py --model_name qwen-7b --model_cache_dir model_cache
  log "处理 NELL23K 原始数据"
  .venv/bin/python scripts/process_raw_data.py --datasets nell23k --output_dir outputs/data --seed 2026
  log "跑 baseline（train + eval，单卡）"
  bash scripts/run_nell23k_full.sh
}

# ---- 主入口 ----
main() {
  local stage="${1:-all}"
  log "========== RecurrentGRIP 服务器脚本开始 stage=$stage =========="
  case "$stage" in
    env)      setup_env ;;
    data)     prepare_data ;;
    smoke)    setup_env; prepare_data; run_smoke ;;
    pilot)    setup_env; prepare_data; run_pilot ;;
    analyze)  run_analyze ;;
    baseline) run_baseline ;;
    all)      setup_env; prepare_data; run_smoke; run_pilot; run_analyze ;;
    *) die "未知 stage: $stage（可选 env|data|smoke|pilot|analyze|baseline|all）" ;;
  esac
  log "========== 完成 stage=$stage =========="
}

main "$@"

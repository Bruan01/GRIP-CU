# WSL2 + RTX 3090 Environment — NELL23K-First v1.1

## Platform Split

- macOS：编辑、Git、AST/Bash/JSON 检查和 NELL23K 纯 Python 数据测试；
- WSL2 + RTX 3090 24GB：Torch/Transformers/PEFT 测试、训练、推理和指标；
- 不复制 `.venv`、模型缓存或运行结果跨平台复用。

## Clone / Update

优先放在 WSL Linux 文件系统：

```bash
cd ~
git clone https://github.com/Bruan01/GRIP-CU.git
git -C GRIP-CU submodule update --init --recursive
cd GRIP-CU/08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
```

已有仓库：

```bash
git pull --ff-only origin main
git submodule update --init --recursive
```

## Environment

```bash
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
```

验收包括 CUDA、RTX 3090、BF16、显存、依赖版本、全部 unittest、PEFT adapter
scope 和 evaluation model CUDA placement。

## Prepare NELL23K

Smoke：

```bash
MAX_TRAIN_QUESTIONS=64 \
MAX_VALIDATION_QUESTIONS=32 \
MAX_TEST_QUESTIONS=64 \
SEED=2026 \
  bash configs/prepare_nell23k.sh
```

数据来自仓库自带：

```text
grip-exp/data/raw_datasets/nell23k/
```

不需要下载 CLEGR。

## Smoke

```bash
RUN_ID=wsl3090_nell23k_smoke_20260829_01 \
  bash configs/run_nell23k_smoke_wsl.sh
```

正式脚本设置：

```text
evaluation_device=cuda
require_cuda=true
use_cache=false
```

因此 CUDA 不可用或 adapter 重载后放置失败时直接报错，不允许 CPU 静默回退。

## Analyze

```bash
RUN_DIR="$(cat results/LAST_NELL23K_SMOKE_RUN.txt)"
RUN_DIR="$RUN_DIR" bash configs/analyze_nell23k.sh
```

## Pilot

只在 Smoke 通过后运行：

```bash
MAX_TRAIN_QUESTIONS=512 \
MAX_VALIDATION_QUESTIONS=128 \
MAX_TEST_QUESTIONS=512 \
  bash configs/prepare_nell23k.sh

RUN_ID=wsl3090_nell23k_pilot_20260829_01 \
  bash configs/run_nell23k_pilot_wsl.sh
```

## Runtime Isolation

每次运行创建新的：

```text
results/runs/<RUN_ID>_nell23k_*/
```

目录已存在时脚本终止。Git 不跟踪 `.venv/`、`model_cache/`、`outputs/`、模型、
checkpoint 或运行结果。

# RecurrentGRIP v1 环境分工：macOS 开发 + WSL2 RTX 3090 执行

## 固定环境边界

| 环境 | 作用 | 不执行 |
|---|---|---|
| macOS 本地 | 编辑代码、维护版本快照、检查 Python AST/Bash/JSON、审阅结果 | 不运行 Qwen 训练，不把 MPS 结果作为论文结果 |
| Windows WSL2 + RTX 3090 24GB | 安装 CUDA PyTorch、运行全部单元测试、tiny PEFT 集成测试、Qwen smoke、正式 pilot | 不修改 Original GRIP |

所有正式运行结果仍写回当前版本的 `results/` 和 `logs/`。`.venv/`、模型缓存和临时输出不属于版本快照。

## 1. 在 macOS 上做静态检查

```bash
cd /Users/mac/CODE/GRIP/ai_research_workflow/08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28
bash configs/check_mac_static.sh
```

该检查不安装 Torch，也不生成项目内 `__pycache__`。

## 2. 将版本快照复制到 WSL2

建议将目录放在 WSL 的 Linux 文件系统，例如：

```text
~/CODE/GRIP/ai_research_workflow/08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28
```

不要长期在 `/mnt/c/...` 上训练，避免 Windows 挂载盘的小文件 I/O 开销。复制时不携带以下目录：

```text
.venv/
model_cache/
outputs/
__pycache__/
```

## 3. WSL2 前置检查

在 Windows 安装支持 WSL2 CUDA 的 NVIDIA 驱动；WSL 内只需要能看到宿主机驱动：

```bash
nvidia-smi
```

然后安装 `uv`，进入版本目录执行：

```bash
bash configs/setup_wsl3090.sh
```

默认环境：

- Python 3.11；
- PyTorch 2.7.1；
- 默认使用 PyPI 对应的 Linux CUDA wheel；
- Transformers 4.56.1；
- PEFT 0.17.1；
- Accelerate 1.10.1。

若需要锁定 PyTorch 官方 CUDA wheel 索引，可覆盖：

```bash
PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
  bash configs/setup_wsl3090.sh
```

脚本同时保留 PyPI 作为依赖索引，避免 `sympy`、`networkx` 等 Torch 依赖解析失败。

## 4. 运行集成测试

```bash
bash configs/test_wsl3090.sh
```

必须确认：

- GPU 名称为 RTX 3090；
- 显存约 24 GiB；
- `torch.cuda.is_available()` 为 true；
- tiny Llama decoder 可重复执行；
- PEFT 仅注入 recurrent executor layer；
- 所有单元测试通过。

## 5. 运行单图 smoke

准备好 CLEGR split 后：

```bash
CLEGR_INPUT=/path/to/recurrent_station_shortest.json \
  bash configs/run_qwen05b_smoke_wsl.sh
```

Smoke 只使用 1 张图、`K=1,2`，用于检查：训练、adapter 保存/加载、CUDA evaluation、cache-free generation 和结果落盘。单图模式自动跳过 shuffled adapter。

## 6. 运行两小时 Pilot

```bash
CLEGR_INPUT=/path/to/recurrent_station_shortest.json \
  bash configs/run_qwen05b_pilot.sh
```

正式脚本固定：

- `--require_cuda true`：CUDA 不可用时立即终止，避免误在 CPU 上运行；
- `--evaluation_device cuda`：重新加载 adapter 后显式把评估模型放到 3090；
- `--bf16 true`；
- Python soft limit 120 分钟；
- Python hard limit 180 分钟；
- WSL GNU `timeout` 再施加 180 分钟操作系统级终止。

每次启动自动创建不可复用的运行目录：

```text
results/runs/YYYYMMDD_HHMMSS_qwen05b_pilot/
```

重复实验通过不同 `RUN_ID` 保存，不覆盖旧结果：

```bash
RUN_ID=seed2027_ablation1 \
CLEGR_INPUT=/path/to/recurrent_station_shortest.json \
  bash configs/run_qwen05b_pilot.sh
```

## 7. 回传结果

每个 `results/runs/<RUN_ID>/` 至少保留：

```text
predictions.jsonl
config.json
environment.txt
run.log
adapters/**/recurrent_manifest.json
analysis/summary.json
analysis/hop_k_accuracy.csv
analysis.log
```

`environment.txt` 已记录 `nvidia-smi`、CUDA/PyTorch 信息和冻结依赖。

不要把 WSL 的 `.venv` 复制回 macOS；两个平台分别建立环境。

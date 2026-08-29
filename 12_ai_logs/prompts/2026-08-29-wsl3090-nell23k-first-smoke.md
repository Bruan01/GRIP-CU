# WSL Codex Prompt — RecurrentGRIP NELL23K-First Smoke

你正在 Windows WSL2 + RTX 3090 24GB 中继续 GRIP-CU 项目。请直接在仓库中执行
下面任务。当前只完成环境集成、测试、NELL23K smoke 和分析；不要启动两小时
Pilot，不要下载 CLEGR。

## 项目边界

```text
Original GRIP（禁止修改）:
13_base_method/grip-exp/

当前可修改版本:
08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/

冻结版本（禁止修改）:
08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/
```

Original GRIP submodule 必须保持：

```text
2835b440bfd2c4de36f0380ae19bc1c22e6cb459
```

## 任务

1. 更新代码：

```bash
git pull --ff-only origin main
git submodule update --init --recursive
```

2. 建立独立分支：

```bash
git switch -c wsl/nell23k-smoke-20260829
```

3. 进入当前版本并执行：

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
bash configs/check_mac_static.sh
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
```

4. 验证 RTX 3090、CUDA、BF16、显存、PEFT adapter scope 和 evaluation CUDA placement。

5. 准备 NELL23K smoke 输入：

```bash
MAX_TRAIN_QUESTIONS=64 \
MAX_VALIDATION_QUESTIONS=32 \
MAX_TEST_QUESTIONS=64 \
SEED=2026 \
  bash configs/prepare_nell23k.sh
```

检查：split 来源、候选答案、train-only graph、BFS structural distance、unknown
hop 和 source SHA256。相同 seed 重新生成到临时文件并核验 SHA256 一致。

6. 运行 smoke：

```bash
RUN_ID=wsl3090_nell23k_smoke_20260829_01 \
  bash configs/run_nell23k_smoke_wsl.sh
```

已有目录时递增 RUN_ID，不覆盖。

7. 分析：

```bash
RUN_DIR="$(cat results/LAST_NELL23K_SMOKE_RUN.txt)"
RUN_DIR="$RUN_DIR" bash configs/analyze_nell23k.sh
```

8. 写入：

```text
logs/WSL_NELL23K_SMOKE_2026-08-29.md
```

必须记录：Git commit、GPU/驱动/CUDA/Torch/Transformers/PEFT、输入统计、测试
结果、RUN_ID、K=1/2 指标、correct/none 指标、用时、峰值显存、错误与修复、是否
建议进入 Pilot。

9. 仅提交代码修复、测试和 Markdown 摘要。不要提交 `.venv`、模型、缓存、
`outputs/` 或原始 `results/runs/`。

## 停止条件

Smoke 和分析完成后停止。不要运行：

```text
configs/run_nell23k_pilot_wsl.sh
configs/prepare_clegr.sh
configs/run_qwen05b_pilot.sh
```

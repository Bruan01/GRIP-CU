# WSL2 RTX 3090 下一步工作

更新日期：2026-08-28

## 当前目标

在 Windows WSL2 + RTX 3090 24GB 上完成 RecurrentGRIP v1 的环境集成、完整测试、CLEGR 精确 hop split 和单图 smoke。**Smoke 通过前不启动两小时 Pilot。**

详细的 Codex 交接 Prompt：

- [`12_ai_logs/prompts/2026-08-28-wsl3090-integration-and-smoke.md`](12_ai_logs/prompts/2026-08-28-wsl3090-integration-and-smoke.md)

## 固定路径

```text
Original GRIP:
13_base_method/grip-exp/

RecurrentGRIP v1:
08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/
```

Original GRIP 作为 submodule 固定在：

```text
2835b440bfd2c4de36f0380ae19bc1c22e6cb459
```

## 待办清单

### A. 仓库与硬件

- [ ] `git pull --ff-only origin main`
- [ ] `git submodule update --init --recursive`
- [ ] 确认 Original GRIP clean 且 commit 正确
- [ ] 确认项目位于 WSL Linux 文件系统，而不是长期运行在 `/mnt/c/...`
- [ ] `nvidia-smi` 识别 RTX 3090 24GB
- [ ] 确认 GNU `timeout` 和 `uv` 可用

### B. 环境与测试

进入：

```bash
cd 08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28
```

依次完成：

```bash
bash configs/check_mac_static.sh
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
```

验收条件：

- [ ] `torch.cuda.is_available()` 为 true
- [ ] GPU 名称为 RTX 3090
- [ ] 显存大于 20 GiB
- [ ] BF16 可用
- [ ] 全部 unittest 通过
- [ ] PEFT 仅注入 recurrent executor layer
- [ ] adapter 重载后的 evaluation model 位于 CUDA

### C. CLEGR 数据

需要找到或生成：

```text
processed_test.json
```

然后运行：

```bash
RAW_INPUT=/实际路径/processed_test.json \
  bash configs/prepare_clegr.sh
```

验收条件：

- [ ] question type 仅为 `StationShortestCount`
- [ ] true hop 由 `edge_index` 上的无向 BFS 重算
- [ ] train/validation 为 1–2 hop
- [ ] test 为 3–4 hop
- [ ] `answer = max(true_hop - 1, 0)`
- [ ] split 无样本泄漏
- [ ] 抽查至少 10 个 BFS/label 样本

### D. 单图 Smoke

```bash
CLEGR_INPUT="$PWD/grip-exp/outputs/data/clegr_reasoning/recurrent_station_shortest.json" \
RUN_ID=wsl3090_smoke_20260828_01 \
  bash configs/run_qwen05b_smoke_wsl.sh
```

若 RUN_ID 已存在，使用 `_02`、`_03`，不得覆盖旧运行。

Smoke 验收条件：

- [ ] Qwen2.5-0.5B 在 CUDA 上训练和评估
- [ ] K=1,2 均产生预测
- [ ] adapter 保存和重载成功
- [ ] `predictions.jsonl`、`environment.txt`、`run.log` 完整
- [ ] 无 OOM、NaN、CUDA error 和 CPU 静默回退
- [ ] 记录用时、峰值显存、K=1/K=2 指标及 correct/disabled 差异

### E. 结果分析与记忆

```bash
RUN_DIR="$(cat results/LAST_SMOKE_RUN.txt)"
RUN_DIR="$RUN_DIR" bash configs/analyze_pilot.sh
```

新增：

```text
08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/logs/
WSL_INTEGRATION_AND_SMOKE_2026-08-28.md
```

记录环境、测试、数据统计、RUN_ID、指标、错误、修复和是否建议进入 Pilot。

## 禁止事项

- 不修改 `13_base_method/grip-exp/`
- 不提交 `.venv`、模型、缓存、outputs 或原始 results
- 不覆盖运行目录
- 不以 CPU 结果作为正式结果
- 不通过删除测试或降低断言解决失败
- 不引入 adaptive halting、gate、frontier loss、routing 或跨图 meta-training
- 不在 Smoke 通过前执行 `configs/run_qwen05b_pilot.sh`

## 本阶段完成定义

满足以下条件后停止并回报：

1. WSL2/RTX 3090 环境验证通过；
2. 全部测试通过；
3. CLEGR split 验证通过；
4. 单图 smoke 完成并分析；
5. 验证 Markdown 已写入；
6. 代码修复和摘要已提交到独立 WSL 集成分支；
7. 尚未启动两小时 Pilot。

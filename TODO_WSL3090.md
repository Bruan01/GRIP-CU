# WSL2 RTX 3090 下一步工作

更新日期：2026-08-29

## 当前目标

在 Windows WSL2 + RTX 3090 24GB 上完成 RecurrentGRIP v1.1 的环境集成、全部
测试、NELL23K 固定 smoke 输入和单图 smoke。**Smoke 通过前不启动两小时 Pilot；
CLEGR 不再是本阶段前置条件。**

完整 Codex 交接 Prompt：

- [`12_ai_logs/prompts/2026-08-29-wsl3090-nell23k-first-smoke.md`](12_ai_logs/prompts/2026-08-29-wsl3090-nell23k-first-smoke.md)

## 固定路径

```text
Original GRIP:
13_base_method/grip-exp/

CLEGR-first frozen snapshot:
08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/

Current NELL23K-first snapshot:
08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/
```

Original GRIP submodule 固定：

```text
2835b440bfd2c4de36f0380ae19bc1c22e6cb459
```

## A. 仓库与硬件

- [ ] `git pull --ff-only origin main`
- [ ] `git submodule update --init --recursive`
- [ ] Original GRIP clean 且 commit 正确
- [ ] 项目位于 WSL Linux 文件系统，不长期运行在 `/mnt/c/...`
- [ ] `nvidia-smi` 识别 RTX 3090 24GB
- [ ] GNU `timeout` 与 `uv` 可用

## B. 环境与测试

```bash
cd 08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29
bash configs/check_mac_static.sh
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
```

验收：

- [ ] `torch.cuda.is_available()` 为 true
- [ ] GPU 为 RTX 3090，显存大于 20 GiB
- [ ] BF16 可用
- [ ] 全部 unittest 通过
- [ ] PEFT 只注入 recurrent executor layer
- [ ] adapter 重载后的 evaluation model 位于 CUDA

## C. NELL23K Smoke 数据

数据已经在版本快照中：

```text
grip-exp/data/raw_datasets/nell23k/
├── train.txt
├── valid.txt
├── test.txt
└── entity2text.json
```

生成固定 64/32/64 输入：

```bash
MAX_TRAIN_QUESTIONS=64 \
MAX_VALIDATION_QUESTIONS=32 \
MAX_TEST_QUESTIONS=64 \
SEED=2026 \
  bash configs/prepare_nell23k.sh
```

验收：

- [ ] train QA 只来自 `train.txt`
- [ ] validation 只来自 `valid.txt`
- [ ] test 只来自 `test.txt`
- [ ] 每题答案都在 candidate relations 中
- [ ] 图只由 `train.txt` 构造
- [ ] structural distance 由 train graph 无向 BFS 重算
- [ ] 不可达样本为 `true_hop=null` 且仍保留
- [ ] `.stats.json` 中 source SHA256、节点、边、关系和 distance buckets 完整
- [ ] 同 seed 重复生成文件哈希一致

## D. NELL23K 单图 Smoke

```bash
RUN_ID=wsl3090_nell23k_smoke_20260829_01 \
  bash configs/run_nell23k_smoke_wsl.sh
```

若目录已存在，使用 `_02`、`_03`，不得覆盖。

验收：

- [ ] Qwen2.5-0.5B 在 CUDA 上训练和评估
- [ ] K=1、K=2 都产生预测
- [ ] correct adapter 与 no-adapter 都产生结果
- [ ] 单图条件下 shuffled control 被明确跳过
- [ ] adapter 保存和重载成功
- [ ] `predictions.jsonl`、`environment.txt`、`input_stats.json`、`run.log` 完整
- [ ] 无 OOM、NaN、CUDA error 或 CPU 静默回退
- [ ] 记录用时、峰值显存、K=1/K=2 和 correct/none 指标

## E. 分析与验证记录

```bash
RUN_DIR="$(cat results/LAST_NELL23K_SMOKE_RUN.txt)"
RUN_DIR="$RUN_DIR" bash configs/analyze_nell23k.sh
```

应生成：

```text
analysis/summary.json
analysis/k_adapter_accuracy.csv
analysis/split_k_adapter_accuracy.csv
analysis/hop_k_accuracy.csv
```

新增：

```text
08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/logs/
WSL_NELL23K_SMOKE_2026-08-29.md
```

记录环境、测试、输入统计、RUN_ID、指标、错误、修复和是否进入 Pilot。

## 禁止事项

- 不修改 `13_base_method/grip-exp/`
- 不修改或覆盖 `v1_fixed_depth_2026-08-28/`
- 不提交 `.venv`、模型、缓存、outputs 或原始运行结果
- 不覆盖已有 RUN_ID
- 不以 CPU 结果作为正式结果
- 不通过删除测试或降低断言解决失败
- 不引入 adaptive halting、gate、frontier loss、routing 或跨图 meta-training
- 不在 Smoke 通过前运行 `configs/run_nell23k_pilot_wsl.sh`
- 本阶段不下载 CLEGR

## 完成定义

满足以下条件后停止并回报：

1. WSL2/RTX 3090 环境验证通过；
2. 全部测试通过；
3. NELL23K 64/32/64 输入验证通过；
4. 单图 smoke 完成并分析；
5. 验证 Markdown 已写入；
6. 必要修复与摘要提交到独立 WSL 集成分支；
7. 尚未启动两小时 Pilot，也未下载 CLEGR。

# NELL23K 验证集自动搜索

本入口复用共享 S1 和 train-only 冻结负样本，仅搜索 S2。不会启动 S1、离线打分或 test 解码，也不会修改当前训练队列。Optuna 的 TPE 算法依据已经完成的验证结果提出后续参数；有限预算下只能得到已尝试配置中的最佳结果，不保证全局最优。

## 搜索空间

| 参数 | 取值 |
| --- | --- |
| 方法 | Soft-Mix、Calibrated |
| 对比损失缩放系数 | 0.1–1.0，对数空间 |
| S2 训练轮数 | 1、3、5、10 |

默认种子 2026，总预算 8 次串行试验。前四次为两种方法各自的缩放系数 1.0 / 0.5、训练 10 轮；之后四次由 TPE 提议。前四次也是全部预算的一部分。每次都从同一个 S1 开始，复用同一份负样本：Soft-Mix 使用固定权重 scale；Calibrated 使用 scale × 原逐题权重。每次生成独立的 `negatives.jsonl`，而不是仅传一个会被逐题权重覆盖的命令行参数。

学习率 1e-3、温度 1、LoRA r=4/alpha=8、梯度累积 512、S1 AdamW 状态继承、answer-only 损失和 mean-logprob 打分保持固定。每次试验训练指定轮数，关闭训练损失阈值早停，结束后完整解码验证集；这不是训练中每个 checkpoint 自动挑最佳。相对于原规程，这是显式的训练轮数搜索变体，不应宣称每个试验都复现了论文的早停。

评分使用自由贪心生成、1000-token 上限和官方答案解析。验证集完整解码 4,951 题，以 `validation.official_harness_em`（4,950 题）选配置。评测输入额外生成仅含 validation 问题的副本。配置选定后，才按实验规程进行最终种子复现和 test 评测；搜索脚本不会自动运行这些步骤。

## 前提与启动

从本项目目录运行。当前论文流程须完成 S1、B1、score，并生成 seed2026 的 calibrated 冻结清单。当前队列若只有 `s1 b1 score`，结束后还需要 freeze。现有 `run_paper_pipeline.sh` 的 freeze 阶段也会生成 Soft-Mix 清单，但自动搜索只需要 calibrated 清单。

```bash
# 安装可选依赖；项目环境没有 pip 时可以使用 uv。
uv pip install --python ../RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp/.venv/bin/python -r requirements-hpo.txt

# CPU 预检，不写产物、不启动 tmux、不占 GPU。
DRY_RUN=1 bash configs/run_validation_hpo.sh

# 当前论文流程退出后，补齐冻结清单；此命令也会进入 tmux。
DATASET=nell23k SEEDS=2026 STEPS=freeze TMUX_SESSION=nell-freeze \
  bash configs/run_paper_pipeline.sh

# 正式搜索，首次运行与中断后恢复使用完全相同的命令。
TMUX_SESSION=nell-validation-hpo N_TRIALS=8 bash configs/run_validation_hpo.sh

tmux attach -t nell-validation-hpo
```

默认固定目录 `results/runs/paper_v2_nell23k/hpo_seed2026/`，总日志为其下 `run.log`，逐次日志为 `trial_000/run.log` 等。可设置 `ROOT`、`SEARCH_DIR`、`S1_ADAPTER`、`CALIBRATED_MANIFEST`。请保持恢复时的目录、输入、预算、代码和 Optuna 版本一致；代码、数据、S1、搜索空间和预算有指纹检查，不匹配会拒绝混用。想更改实验定义须使用新的 `SEARCH_DIR` 并记录原因。

## 恢复与结果

- SQLite `study.sqlite3` 保留参数与评分；中断的 RUNNING 试验先恢复，不新建替代试验，也不重置总预算。
- 每 10 个 optimizer step 保存 checkpoint，保留最近 2 个，自动从 `trainer_listed/checkpoint-*` 恢复。终态 adapter 位于 `trial_NNN/listed/adapter/`；已完成时加 `--skip_train`，只续写尚未解码的验证结果。
- 当前论文流程的 `pipeline.lock` 和搜索自己的 `search.lock` 都采用排他锁；当前流程还在运行时搜索直接报错，不会排队自动启动。其他不使用这些锁的 GPU 任务仍需要人工避开。
- 训练失败时停止搜索并保留同一 trial，错误记录在 `trials.json` 的 `attempt_errors`。修复环境问题后重跑同一命令即可。确定不可恢复的配置需要单独处理并记录，脚本不会偷偷跳过。
- `best.json` 提供总体及各方法的最佳配置；`trials.json`、`VARIANTS.md` 保留全部试验，包括未完成的试验。相同分数选择更早的试验。`best.json` 随搜索更新，只有整个预算完成后才视为最终选择。
- 搜索后仍需验证所选配置的跨种子稳定性。自动搜索本身不是超过论文 87.74% 的证据，也不能消除对验证集多次选择带来的过拟合风险。

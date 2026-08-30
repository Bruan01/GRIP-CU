# RecurrentGRIP 远程拉取、现有结果复核与下一步准备（2026-08-30）

## 1. 拉取结果

在分支 `wsl/nell23k-smoke-20260829` 执行 `git pull --ff-only`，从 `3ce6df3` 快进到：

```text
6809e79005c012c50a8435209c0276accc5ed68c
fix: make test_wsl3090.sh PYTHON overridable
```

远程只有 `main` 和 `wsl/nell23k-smoke-20260829` 两个分支；没有其他未合并的结果分支。
本次远程更新只改动一行环境脚本，没有新增 v1.1.1 run artifacts。

## 2. 结果可用性审计

### 已存在

- v1.1 WSL smoke `_06` 的人工归档日志；
- 64/32/64 NELL23K 输入及 stats；
- v1.1.1 的静态 context selection 审计；
- v1.1.1 runner、cross-run audit、metrics 和 state-dynamics 分析代码。

### 不存在

远程仓库和当前 Mac clone 中尚不存在 v1.1.1 的真实 GPU 产物；但 WSL 本地已经完成：

```text
results/runs/wsl3090_nell23k_diag_cross_20260830_01_nell23k_diagnostic_cross/
```

该目录预计包含 `predictions.jsonl`、`cross_run_audit.json`、analysis 文件和两个 context
manifests。根 `.gitignore` 使用 `**/results/*`，所以这些 WSL 本地文件没有进入 Git，Mac
执行 `git pull` 也不会得到它们。当前应先从该现有 run 导出小型审计产物，不需要重跑实验；
在导出内容推送前，仍不能对 v1.1.1 做 Go/Pivot/Stop 判定。

## 3. v1.1 smoke 的定量复核

输入为 96 个 evaluation questions、10 个 relation candidates；训练深度为 `K_train=2`。

| adapter | K=1 | K=2 | K2-K1 |
|---|---:|---:|---:|
| correct | 29/96 (30.21%) | 3/96 (3.13%) | -26/96 (-27.08 pp) |
| none | 25/96 (26.04%) | 1/96 (1.04%) | -24/96 (-25.00 pp) |

### 3.1 第二次 recurrence 是系统性破坏，不是小波动

correct adapter 在 K1/K2 合计只解出 31 个不同问题。因此可以恢复配对转移：

| transition | count |
|---|---:|
| K1 correct → K2 correct | 1 |
| K1 correct → K2 wrong | 28 |
| K1 wrong → K2 correct | 2 |
| K1 wrong → K2 wrong | 65 |

K2 只新增解决 2 题，却破坏 28 题，净损失 26 题。K2 accuracy 也低于 10-candidate 的名义
均匀随机水平。none control 同样下降 25.00 pp，说明故障至少部分来自共享 decoder block 的
重复执行，而不是仅由 graph adapter 引起。

### 3.2 旧结果不能证明 graph-specific storage

旧 sampler 先列出约 20,799 个 node declarations，再列 edge facts，却对完整列表做
`[:256]`。因此 256 个 context slots 基本全部是 node declarations，不能支持“relation edge
facts 已参数化写入”的结论。correct-none 差值仅为：

- K1：`4/96 = 4.17 pp`；
- K2：`2/96 = 2.08 pp`。

没有逐题原始 prediction 和 split-level 配对统计时，这两个差值只能视为待验证信号。

### 3.3 `K_train/K_eval mismatch` 不是唯一解释

v1.1 实际 `K_train=2`，所以 K2 是 depth-matched 条件，却仍从 30.21% 降到 3.13%。这使
“只是测试深度与训练深度不匹配”成为不充分解释。仍需加入 `K_train=1`，才能判断：

- adapter 是否只对某个训练深度有效；
- 重复 block 是否独立于训练深度持续漂移；
- K1 的较高分是否来自提前读取 decoder 表示，而非递归推理。

### 3.4 当前 claim boundary

可支持：

> 工程链路可运行；当前 direct recurrence 的第二步表现出强破坏性，必须分离 storage input、
> adapter signal 与 execution stability。

不支持：

- RecurrentGRIP 提升 NELL23K；
- K 对应 graph hop；
- graph facts 已进入参数记忆；
- recurrence 的成本或精度优于 Original GRIP；
- 任何 SOTA/刷榜结论。

## 4. 已完成的下一步准备

新增 compact artifact exporter：

- `configs/export_diagnostic_cross_artifacts.py`；
- `configs/export_diagnostic_cross_artifacts.sh`；
- `tests/test_export_diagnostic_artifacts.py`。

并将 exporter 接入 `run_nell23k_diagnostic_cross_wsl.sh`。完整 run 仍保留在 ignored results
目录，但以下内容会自动导出到 Git 可跟踪目录：

- config、input stats、environment；
- cross-run audit；
- train_k1/train_k2 context manifests；
- 全部 analysis JSON/CSV；
- 去除完整 hidden-state vectors 的逐题 `predictions_audit.jsonl`；
- 含 SHA256、大小、commit、prediction count 和 redaction 说明的 artifact manifest。

导出器拒绝：缺文件、cross audit 非 pass、prediction/summary count 不一致和覆盖已有导出目录。

## 5. 下一步决策

当前状态：**导出并回传已经完成的 diagnostic cross；不重跑、不进入 Pilot、不开始 v1.2。**

下一步 WSL 命令和结果回传方式见：

```text
08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29/NEXT_STEP_WSL_2026-08-30.md
```

只有从 WSL 现有 run 收到真实的 768 条 2×2 cross prediction 后，才进行正式的 Go/Pivot/Stop 机制判断。

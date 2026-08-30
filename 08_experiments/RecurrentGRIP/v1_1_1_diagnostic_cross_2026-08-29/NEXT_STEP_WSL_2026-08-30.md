# RecurrentGRIP v1.1.1 — WSL 下一步执行单

## 当前状态（2026-08-30）

- 远程分支：`wsl/nell23k-smoke-20260829`；
- 已拉取远程提交：`6809e79005c012c50a8435209c0276accc5ed68c`；
- 该提交只修复 `configs/test_wsl3090.sh`，允许使用现有 conda Python；
- WSL 本地已经存在 v1.1.1 run：
  `results/runs/wsl3090_nell23k_diag_cross_20260830_01_nell23k_diagnostic_cross/`；
- 该目录被根 `.gitignore` 的 `**/results/*` 排除，因此 Mac/远程仓库尚未收到 prediction、
  summary 或 state-dynamics；
- 当前优先导出现有 run，不重跑、不启动 512/128/512 Pilot、不下载 CLEGR。

## 为什么必须先运行 diagnostic cross

v1.1 smoke 使用 node-first context 加 `max_context_samples=256`，在 20,799 个 node
声明之后才出现 edge，因此该 adapter 的 graph context 基本没有 relation edge facts。
同时，smoke 只训练了 `K_train=2`：

| control | K=1 | K=2 | 变化 |
|---|---:|---:|---:|
| correct | 29/96 = 30.21% | 3/96 = 3.13% | -26 题 / -27.08 pp |
| none | 25/96 = 26.04% | 1/96 = 1.04% | -24 题 / -25.00 pp |

correct adapter 的 31 个去重已解问题结合 K1/K2 正确数可推出：K1/K2 同时正确 1 题、
K1→K2 被改错 28 题、K1 错而 K2 新解 2 题、两者都错 65 题。这个结果说明第二次执行
主要在破坏答案，但旧 context 无法区分 storage failure 与 executor instability。

## WSL 现有结果导出命令

先拉取包含导出器的最新提交，然后直接导出现有 run，不重新训练：

```bash
cd ~/GRIP-CU
git switch wsl/nell23k-smoke-20260829
git pull --ff-only origin wsl/nell23k-smoke-20260829

VERSION_DIR=08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29
RUN_DIR="$PWD/$VERSION_DIR/results/runs/wsl3090_nell23k_diag_cross_20260830_01_nell23k_diagnostic_cross"

cd "$VERSION_DIR"
export PYTHON="$(command -v python)"

RUN_DIR="$RUN_DIR" \
PYTHON="$PYTHON" \
  bash configs/export_diagnostic_cross_artifacts.sh
```

导出器检查：

1. `cross_run_audit.json` 必须为 `status=pass`；
2. prediction count 必须与 cross audit 和 summary 一致；
3. 两个真实 context manifests 必须存在；
4. accuracy、transition、output quality 和 hidden-state dynamics 文件必须齐全；
5. 自动将小型审计产物写入 `09_results_analysis/artifacts/RecurrentGRIP_v1_1_1/`；
6. 完整 hidden-state vectors 不进入 Git，但聚合 state dynamics 保留。

## 结果回传

```bash
cd ~/GRIP-CU
git status --short
find 09_results_analysis/artifacts/RecurrentGRIP_v1_1_1 -maxdepth 3 -type f -print

git add 09_results_analysis/artifacts/RecurrentGRIP_v1_1_1
git commit -m "results: add RecurrentGRIP v1.1.1 diagnostic cross"
git push origin wsl/nell23k-smoke-20260829
```

不要提交：

- `results/runs/.../trainer/`；
- adapters 或模型权重；
- 完整 `step_pooled_hidden_states`；
- Hugging Face cache、conda 环境或 NELL23K 原始下载目录。

## 收到结果后的固定分析顺序

1. **完整性**：`cross_run_audit.status=pass`、768 条 prediction、每个矩阵格 96 条；
2. **Storage signal**：validation/test 分开比较 correct vs none，并做逐题配对 transition；
3. **Depth compatibility**：比较 train1/eval1、train1/eval2、train2/eval1、train2/eval2；
4. **Destruction vs gain**：报告 `K1 correct→K2 wrong` 与 `K1 wrong→K2 correct`；
5. **Output failure**：检查 empty、candidate-out、EOS、generated-token 分布；
6. **State drift**：检查 norm ratio、consecutive cosine 和 relative delta 是否与错误转移同向；
7. **决策**：只给出 Go / Pivot / Stop，不从该 diagnostic 声称 SOTA 或论文主结果。

## 下一版触发条件

- **Go**：correct 相对 none 在 validation/test 方向一致，depth-matched K2 有恢复，且 K2 新解
  不只是偶然个例；再创建 v1.2 并测试稳定化机制。
- **Pivot**：adapter 有信号，但 K2 仍伴随明显 state drift；v1.2 按 outer norm → input recall →
  residual scale → gate 的顺序做单变量消融。
- **Stop recurrence**：correct≈none、train2/eval2 仍与 train1/eval2 同幅塌缩、K2 几乎只改坏
  K1 正确题；保留负结果，将主线转回 GRIP storage/routing/continual update。

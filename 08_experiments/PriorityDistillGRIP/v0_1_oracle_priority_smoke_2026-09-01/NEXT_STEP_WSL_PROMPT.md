# Prompt for WSL Codex

```text
你现在位于 GRIP-CU 仓库的 WSL2 + RTX 3090 环境。请执行 PriorityDistill-GRIP v0.1 的 one-seed oracle priority smoke，不修改 Original GRIP，也不实现 learned prioritizer。

实验目录：
08_experiments/PriorityDistillGRIP/v0_1_oracle_priority_smoke_2026-09-01/

必须先阅读：
1. README.md
2. PROJECT_MEMORY.md
3. SOURCE_BASELINE.md
4. configs/oracle_priority_smoke.json

执行顺序：
1. git status --short --branch，保留已有文件，不清理无关内容。
2. 创建/激活 Python venv，安装 requirements-wsl.txt；确认 RTX 3090 可见。
3. 运行 bash configs/check_wsl_runtime.sh。
4. 使用固定 RUN_ID=wsl3090_priority_distill_smoke_01 运行 bash configs/run_wsl_smoke.sh。
5. 检查五个 method/seed_42 是否都有 run_summary.json、validation/test predictions 和 token_budget_audit.json。
6. 打开 REPORT.md 与 suite_summary.json，特别核对：
   - Stage-1 input token 是否近似匹配；
   - oracle 是否超过 answer_only、more_qa、random_path、all_paths；
   - 3/4-hop 是否改善；
   - inference_graph_access 是否全部为 false；
   - 是否有 OOM、NaN、截断或异常 optimizer-step 差异。
7. 调用科研结果分析/experiment-audit/result-to-claim skill（若可用），写 RESULTS_ANALYSIS.md；单 seed 只能写 preliminary conclusion。
8. 如果结果为 PRELIMINARY_GO，先提交 one-seed 结果，再建议是否跑 full seeds；如果 PRELIMINARY_STOP，停止本方向，不加入 scorer、DPO 或新模块。
9. 不覆盖本版本，任何代码修复创建 v0_1_1 或更高版本目录。
10. 提交时包含指标、报告、环境、预测与分析；adapter_model.pt 保持忽略。
```

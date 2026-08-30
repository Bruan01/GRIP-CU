# 给 WSL Codex 的下一步 Prompt

```text
你在 WSL2 + RTX 3090 24GB 上工作，仓库为 GRIP-CU，分支以当前远程分支为基线。

目标：实现并运行独立实验
08_experiments/FactorGRIP/v0_1_candidate_energy_probe_2026-08-30/
不得修改 13_base_method/grip-exp/，不得覆盖任何 RecurrentGRIP 版本或已有 run。

先阅读：
1. 09_results_analysis/2026-08-30_recurrentgrip_v1_1_1_diagnostic_cross/REPORT.md
2. 08_experiments/FactorGRIP/v0_1_candidate_energy_probe_2026-08-30/README.md
3. RecurrentGRIP v1.1.1 的 config、runner、tests 和现有结果结构。

任务：
1. 从 v1.1.1 快照复制最小必要代码到本版本 grip-exp/，记录 SOURCE_BASELINE.md 和 PATCH_MANIFEST.md。
2. 新增 inference-only candidate scoring：对每题 10 个 relation candidates 计算长度归一化 sequence log-likelihood；优先一次 batch 完成 10 candidates。
3. 保留 free generation，增加 candidate-constrained generation；不得使用 target 构造输入。
4. 支持 correct/none；若当前 adapter 结构允许，增加 shuffled/wrong-adapter control，否则在结果中明确 unavailable。
5. 保存逐候选分数、预测、正确性、延迟、峰值显存、environment 和不可覆盖的 RUN_ID。
6. 添加纯 Python 单测：候选排序、长度归一化、target 不泄漏、candidate order permutation、resume/overwrite guard。
7. 先运行 mac/static tests，再运行 NELL23K 64/32/64、Qwen2.5-0.5B probe。
8. validation 冻结 temperature/normalization 后再输出 test；生成 paired McNemar 和 bootstrap CI。
9. 不启动 7B、不运行其他数据集、不实现 FactorGRIP experts。
10. 完成后提交代码和 compact results 到当前工作分支，汇报 commit、RUN_ID、准确率表、correct-none delta、score-generation delta、GPU wall time 和是否通过 Go gate。
```

# GRIP-CU

AI-assisted research workflow for GRIP and RecurrentGRIP.

这是一套适合计算机 / AI / 机器学习研究的科研项目模板。

核心流程：

Problem
→ Related Work
→ Research Gap
→ Benchmark
→ Baseline
→ Hypothesis
→ Method
→ Experiment
→ Analysis
→ Conclusion
→ Paper

真正的科研不是线性的，而是：

Method v1
→ Experiment v1
→ Failure Analysis
→ Method v2
→ Experiment v2
→ ...
→ Final Method

---

## 目录说明

### 01_problem
定义研究问题。回答：到底要解决什么问题？为什么重要？

### 02_related_work
检索、阅读、分类已有工作。不要只做论文摘要，要关注任务、方法、数据集、指标、局限。

### 03_gap_analysis
从 Related Work 中提炼真正的 Research Gap，并明确“别人已经做了什么、还没做什么、你的机会在哪里”。

### 04_benchmark
确定数据集、benchmark、数据切分、评价协议和指标。必须与研究问题一致。

### 05_baseline
复现强 baseline 和官方结果，建立可信实验起点。

### 06_hypothesis
把“我觉得这样可能有效”写成可被实验验证或证伪的科研假设。

### 07_method_versions
记录算法从 v0 到 final 的所有版本，禁止只保留最终版本，以便复盘。

### 08_experiments
独立管理主实验、消融、超参数、效率、鲁棒性和错误分析。

### 09_results_analysis
分析为什么有效、在哪些场景有效、为什么失败，不只看一个最终数字。

### 10_conclusion
形成结论、贡献、局限和未来工作。

### 11_paper
最终论文写作材料，包括 Introduction、Related Work、Method、Experiments、Conclusion 等。

### 12_ai_logs
保存 AI 辅助科研的重要对话、提示词、检索结论、代码分析记录和决策记录。

---

## AI 在科研中的正确角色

AI 可以帮助：
- 论文检索与横向比较
- 研究问题拆解
- Gap 分析
- Benchmark 调研
- Baseline 代码阅读
- 实验设计
- 方法 brainstorm
- 实验结果分析
- 论文结构与语言优化
- LaTeX / 图表 / 代码辅助

AI 不应该替代：
- 对论文原文的核验
- 对实验真实性的判断
- 对创新性的最终判断
- 对实验数据的真实性负责
- 虚构结果、引用或结论

---

## 最关键的五个问题

每个 idea 在投入大量代码前，先回答：

1. Problem：到底解决什么问题？
2. Gap：别人为什么还没有解决？
3. Method：你的方法为什么应该有效？
4. Benchmark：在哪个公认任务上证明？
5. Evidence：能不能稳定超过强 baseline？

如果这五个问题都清楚，再进入正式开发。

---

## 当前研究主线：EntityConstrained-GRIP

更新日期：2026-09-04

当前问题：

> train-KG-only 全局实体约束能否把冻结 GRIP checkpoint 中未被自由生成读出的 latent entity preference 转化为 semantic exact match，而不是只提高输出合法率？

当前状态：

- PriorityDistill-GRIP 的稳定涨点 claim 已停止；
- E03 已修订为 **validation-only D0 artifact reuse + D1 global trie**；
- primary 为 validation-selected direct seed43/44，More-QA seed42 为 reference，joint seed43/44 为 secondary；
- Phase-A runner 不打开 test，并输出 D0→D1 invalid/valid-wrong/correct error transitions；
- gate 要求两个 direct checkpoints 平均 canonical EM `≥ +2 pp`、每 checkpoint raw/canonical 非负、novel-composition 平均下降不超过 `1 pp`；
- D2 只有 `PRELIMINARY_GO_D2` 后才运行，D3 始终 diagnostic-only；
- 完整静态门禁、协议残留搜索和 Git 边界审计已通过，当前状态为 `READY_TO_PUSH_VALIDATION_ONLY_E03`；
- 当前仍未 commit、未 push、未运行 WSL GPU；机制通过后才返回官方 Qwen2.5-7B、LoRA rank/alpha `4/8`、三随机种子 Phase B。

当前执行入口：

1. [`TODO_ENTITY_CONSTRAINED_GRIP.md`](TODO_ENTITY_CONSTRAINED_GRIP.md)
2. [`08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/README.md`](08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/README.md)
3. [`08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/EXPERIMENT_PLAN.md`](08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/EXPERIMENT_PLAN.md)
4. [`08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/PROJECT_MEMORY.md`](08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/PROJECT_MEMORY.md)
5. [`08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/NEXT_STEP_WSL_PROMPT.md`](08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/NEXT_STEP_WSL_PROMPT.md)
6. [`08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/artifacts/setup_audit.json`](08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/artifacts/setup_audit.json)

PriorityDistill、StructuredLoRA、RecurrentGRIP 的历史状态和负结果继续保留，不删除、不覆盖。

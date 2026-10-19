# GRIP-CU

AI-assisted research workflow for GRIP candidate-relation learning.

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

## 当前研究主线：GRIP 候选关系学习

更新日期：2026-10-19

本项目当前只保留三条可比较路径：

1. **Original GRIP/B1**：原始 GRIP 的 generation-only 基线；
2. **Random-9 + InfoNCE**：同一训练预算下随机采样 9 个候选负样本；
3. **ErrorMemory-9 + InfoNCE**：从冻结 B1 的训练集错误和候选分数违例中构造 9 个边界负样本。

两套首轮数据集为 NELL23K 和 FB15K237-2。三条路径保持原始 GRIP 的 graph-context、LoRA 和生成目标不变，只在 Stage 2 的候选关系学习阶段加入对比约束。

当前执行入口：

1. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)
2. [`04_benchmark/README.md`](04_benchmark/README.md)
3. [`08_experiments/Hard_Negative_Contrastive_GRIP/README.md`](08_experiments/Hard_Negative_Contrastive_GRIP/README.md)
4. [`08_experiments/Hard_Negative_Contrastive_GRIP/configs/run_error_memory_wsl.sh`](08_experiments/Hard_Negative_Contrastive_GRIP/configs/run_error_memory_wsl.sh)

当前原则：先完成两个数据集的完整 task、B1 错误/分数冻结、Random-9 与 ErrorMemory-9 的配对训练和审计，再报告候选关系 MRR/Hits、free-generation EM、wrong-in-list、OOV 和候选有效率。没有实验结果前不宣称提升。

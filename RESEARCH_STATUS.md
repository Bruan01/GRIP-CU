# RecurrentGRIP Research Status

更新日期：2026-08-28

## 当前主线

**RecurrentGRIP: Compiling Graphs into Executable Parametric Programs for Closed-Book Graph Reasoning**

核心问题不是继续优化 GRIP 的训练配方，而是区分：

- 图是否成功写入参数（storage）；
- 参数化图是否能被模型逐步调用（retrieval）；
- 被调用的图知识是否能形成可重复执行的多跳计算（execution）。

## 当前主张

RecurrentGRIP 将模型拆分为：

1. 共享问题编码器；
2. 跨图共享的递归图执行器；
3. 图专属 LoRA 参数记忆；
4. 答案解码器与可选动态停止头。

图专属 LoRA 定义“执行哪张图”，共享递归模块学习“如何执行一步图传播”。推理时不提供原图。

## 当前阶段

- [x] 代码结构初步检查
- [x] 研究问题确定
- [x] 三个跨学科方向比较
- [x] 选择 RecurrentGRIP
- [x] 初步 novelty check
- [x] 方法与实验设计 v0.1
- [ ] 原始 GRIP baseline 完整复现
- [x] Fixed-Depth v1 文件级实现计划
- [x] CLEGR 数据 schema 修复与精确 hop split代码实现
- [ ] CLEGR 两小时 pilot
- [ ] Pilot 决策门
- [ ] 完整实现（当前已完成 Fixed-Depth v1 scaffold）
- [ ] 六数据集实验
- [ ] 机制分析与论文写作

## 当前代码版本

- Original GRIP：`13_base_method/grip-exp/`，保持 clean，不直接修改；
- RecurrentGRIP v1：`08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28/`；
- 开发环境：macOS，仅做代码、版本管理和静态检查；
- 执行环境：Windows WSL2 + RTX 3090 24GB，负责 ML 集成测试、Qwen smoke 和正式 Pilot；
- 已完成：独立代码快照、CLEGR metadata/hop split、fixed-depth executor、adapter controls、测试 scaffold、平台脚本和显式 CUDA evaluation placement；
- 待完成：在 WSL2 上运行 Torch/Transformers/PEFT 集成测试、Qwen smoke test 与两小时 GPU pilot。

## 立即执行顺序

> WSL 执行端的当前 TODO 与完整 Codex Prompt 见 [`TODO_WSL3090.md`](TODO_WSL3090.md)。

1. 冻结原始 GRIP baseline，不修改其默认行为。
2. 核验 NELL23K 与 CLEGR 当前复现结果。
3. 构造 CLEGR 精确 hop split，并检查所有路径捷径。
4. 实现最小固定深度 recurrent executor。
5. 将独立版本复制到 WSL2 Linux 文件系统，建立 RTX 3090 CUDA 环境。
6. 先运行单图 smoke，再运行不超过两小时的 pilot。
7. 只有通过预设成功条件，才实现动态停止、frontier probing 和六数据集扩展。

## 核心文档

- `01_problem/problem_statement.md`
- `02_related_work/literature_map.md`
- `03_gap_analysis/gap_matrix.md`
- `04_benchmark/protocol.md`
- `05_baseline/baseline_registry.md`
- `06_hypothesis/hypotheses.md`
- `07_method_versions/v1_first_core_idea/recurrentgrip_design.md`
- `07_method_versions/v1_first_core_idea/IMPLEMENTATION_PLAN.md`
- `08_experiments/PILOT_PLAN.md`
- `08_experiments/experiment_matrix.md`
- `11_paper/paper_outline.md`

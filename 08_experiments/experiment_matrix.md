# Experiment Matrix

更新日期：2026-08-28

## E0：Baseline Reproduction

| ID | 内容 | 数据集 | 输出 |
|---|---|---|---|
| E0.1 | Base LLM | NELL23K | 泄漏率 |
| E0.2 | Original GRIP | NELL23K | EM/F1/Hit |
| E0.3 | Original GRIP | CLEGR subset | hop 分桶结果 |
| E0.4 | Original GRIP | Scene Graph subset | 基本复现 |

## E1：Pilot

| ID | 内容 | 关键变量 | 判定 |
|---|---|---|---|
| E1.1 | Original GRIP | K=1 | 主基线 |
| E1.2 | More-QA GRIP | 同训练数据 | 排除数据量 |
| E1.3 | RecurrentGRIP | K=1–5 | recurrence scaling |
| E1.4 | Shuffled adapter | 错误图 LoRA | 图记忆必要性 |

## E2：主结果

六数据集比较：

- Original GRIP；
- More-QA GRIP；
- GRIP-Deep；
- ComputeMatch；
- Self-Refine；
- RecurrentGRIP；
- Graph Context / KG-RAG；
- Oracle Path。

分别报告：

- ID；
- Length OOD；
- Relation Composition OOD；
- Label OOD；
- Paraphrase OOD。

## E3：机制

| ID | 实验 | Claim |
|---|---|---|
| E3.1 | hop–recurrence heatmap | H1 |
| E3.2 | frontier linear probe | H2 |
| E3.3 | remaining-distance probe | H2 |
| E3.4 | correct-to-wrong patch | H4 |
| E3.5 | wrong-to-correct patch | H4 |
| E3.6 | step deletion | H4 |
| E3.7 | step permutation | H4 |
| E3.8 | adapter swap | H5 |

## E4：消融

- shared vs unshared；
- executor layer position；
- 1 block vs 2 blocks；
- step embedding；
- update gate；
- fixed vs adaptive halt；
- answer-only loss；
- depth loss；
- stability loss；
- frontier-supervised auxiliary version；
- LoRA target modules；
- LoRA rank。

## E5：效率

报告：

- adapter size；
- trainable parameters；
- internalization time；
- average recurrence；
- latency；
- FLOPs；
- peak GPU memory；
- accuracy/FLOP Pareto curve。

## E6：鲁棒性和失败分析

- 大分支因子；
- 高度节点；
- 环；
- 多答案；
- 关系歧义；
- entity alias；
- adapter quantization；
- rank truncation；
- 过度递归；
- 图写入失败与执行失败的区分。

## E7：统计显著性

正式主结果：

- 至少 3 个随机种子；
- 配对 bootstrap 或适合任务的配对检验；
- 报告均值、标准差、置信区间；
- 同时报告效果量，避免只看 p-value。

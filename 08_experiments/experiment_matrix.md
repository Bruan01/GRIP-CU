# Experiment Matrix

更新日期：2026-08-29

## E0：Baseline Reproduction

| ID | 内容 | 数据集 | 输出 |
|---|---|---|---|
| E0.1 | Base LLM | NELL23K | Accuracy/EM |
| E0.2 | Original GRIP | NELL23K | Accuracy/EM、延迟、显存 |
| E0.3 | Original GRIP | CLEGR subset | 延后：严格 hop 分桶 |
| E0.4 | Original GRIP | 其他 GRIP 数据集 | 主结果扩展 |

## E1：NELL23K-First Pilot

| ID | 内容 | 关键变量 | 判定 |
|---|---|---|---|
| E1.1 | RecurrentGRIP smoke | K=1,2；64/32/64 QA | 运行链与 CUDA 正确性 |
| E1.2 | Adapter control | correct vs none | 图参数记忆必要性 |
| E1.3 | RecurrentGRIP pilot | K=1–4；512/128/512 QA | recurrence scaling |
| E1.4 | Structural-distance buckets | train-graph BFS distance | K 与图结构距离的相关性 |
| E1.5 | Original GRIP fair baseline | 同模型、LoRA、QA、seed | 排除训练预算差异 |

NELL23K 当前是一个图，因此 E1 不强制 shuffled-adapter；跨图数据集阶段再加入
adapter swap/shuffle。

## E2：CLEGR Mechanism Confirmation

| ID | 内容 | 关键变量 | Claim |
|---|---|---|---|
| E2.1 | StationShortestCount | true hop × K | 严格 K↔hop |
| E2.2 | correct/shuffled/none | graph adapter identity | 图专属参数程序 |
| E2.3 | step trace/probe | recurrence state | 中间计算轨迹 |

只有 E1 出现可信 recurrence 信号后启动 E2。

## E3：主结果

扩展比较：Original GRIP、More-QA GRIP、ComputeMatch、Self-Refine、
RecurrentGRIP、Graph Context/KG-RAG 和 Oracle Path。数据集优先从 NELL23K、
CoDEx-M、FB15k-237、WN18RR 逐步扩展，而不是一次性运行全部组合。

## E4：机制与消融

- shared vs unshared executor；
- executor layer position；
- K_train × K_eval；
- correct/none/shuffled adapter；
- step hidden-state probe；
- step deletion/permutation；
- LoRA target modules 与 rank；
- fixed vs adaptive halt（仅 fixed-depth 成功后）。

## E5：效率与统计

报告 adapter size、trainable parameters、internalization time、latency、FLOPs、
peak GPU memory 和 accuracy/FLOP Pareto。正式主结果至少 3 seeds，并使用配对
bootstrap 或适当的配对检验，报告均值、标准差、置信区间和效果量。

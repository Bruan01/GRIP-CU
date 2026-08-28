# Research Questions

更新日期：2026-08-28

## RQ1：Storage

图专属 LoRA 中可被可靠恢复的内容是什么？

- 一跳事实；
- 关系类型；
- 多跳可达性；
- 路径组合；
- 图的局部转移结构。

## RQ2：Execution

共享递归模块重复调用图专属 LoRA 时，是否表现为逐跳图执行？

检验标准：最优递归次数与真实最短路径长度正相关。

## RQ3：Intermediate State

第 \(k\) 轮隐状态是否编码第 \(k\) 跳 frontier、候选节点集合或剩余距离？

检验标准：冻结模型后的线性 probe，以及对隐状态的因果 patching。

## RQ4：Length Extrapolation

只使用 1–2 hop 或 1–3 hop 任务训练时，测试阶段增加 recurrence 是否能够改善 3–8 hop 问题？

## RQ5：Composition

RecurrentGRIP 能否组合训练中未出现过的 relation sequence，而不是复现已见路径模板？

## RQ6：Graph-Size Generalization

共享执行器能否从小图迁移到更大的图，而图专属 LoRA 只负责提供图内容？

## RQ7：Cost Boundary

与普通 GRIP、更多 CoT、显式图上下文和 KG-RAG 相比，RecurrentGRIP 在准确率、延迟、FLOPs、参数存储上的边界在哪里？

## RQ8：Failure Boundary

在哪些场景中 recurrence 不产生有效图执行？重点分析：

- 高分支因子；
- 多答案集合；
- 关系歧义；
- 环；
- 过度递归；
- 图记忆本身写入失败。

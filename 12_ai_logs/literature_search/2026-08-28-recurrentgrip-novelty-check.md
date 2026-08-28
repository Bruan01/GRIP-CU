# Novelty Check Log: RecurrentGRIP

日期：2026-08-28

## Proposed Method

将图写入图专属 LoRA，并使用跨图共享的 recurrent executor 在推理时无图输入条件下重复执行该参数化图算子。

## Core Claims

| Claim | 初步 novelty | 最近邻风险 |
|---|---|---|
| 内化图的 executable memory 定义 | 高 | GRIP 与 storage–retrieval gap |
| shared executor + graph-specific LoRA | 中高 | recurrent Transformer、adapter composition |
| test recurrence 支持 unseen-hop 外推 | 中高 | recurrent graph reachability |
| frontier + causal patching 机制证据 | 高 | neural algorithmic reasoning、knowing–using gap |

## Closest Prior Work

| 工作 | 重叠 | 关键差异 |
|---|---|---|
| GRIP | 图写入 LoRA、闭卷推理 | 没有共享递归执行和 hop-aligned trajectory |
| Storage–Retrieval Gap in Graph Reasoning | 参数图记忆选择与组合 | 不以 recurrent graph program 作为核心机制 |
| Depth-Recurrent Transformer | 共享深度计算 | 没有图专属参数记忆 |
| Recurrent Transformers Learn to Reason | recurrence 与推理 | 没有图编译、闭卷图执行 |
| Recurrent graph reachability | 多步图算法 | 推理时通常有显式图结构 |
| Knowing–Using Gap | 存储与使用错位、因果干预 | 不是图专属参数程序 |

## Overall Assessment

- 初步分数：7.5–8/10；
- 建议：Proceed with a tightly defined mechanism claim；
- 关键差异：the graph is compiled into parameters before inference, and a shared recurrent executor repeatedly operates on that graph-specific parameter memory without graph access；
- 最大风险：如果只有准确率提升而没有 frontier 和 causal evidence，会被视为 recurrence 应用论文。

## 官方检索入口

- GRIP：https://arxiv.org/abs/2511.07457
- Storage–Retrieval Gap：https://arxiv.org/abs/2608.25489
- Knowing–Using Gap：https://arxiv.org/abs/2607.08393
- Depth-Recurrent Transformer：https://openreview.net/forum?id=Q7boRE3kBy
- Recurrent Transformers Learn to Reason：https://openreview.net/forum?id=Au7WqYeoHb
- RANK：https://arxiv.org/abs/2608.11055
- Recurrent Attention for Graph Reasoning：https://arxiv.org/abs/2602.04445

## 投稿前复查

投稿前必须重新检索最近六个月的：

- in-parameter graph memory；
- recurrent graph reasoning；
- graph-specific LoRA；
- test-time depth graph reasoning；
- executable parameter memory；
- adapter composition for graph QA。

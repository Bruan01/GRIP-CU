# RecurrentGRIP Gap Matrix

更新日期：2026-08-28

## 核心 Gap 表述

Existing methods can store graph knowledge in language-model parameters or perform recurrent reasoning over explicitly provided structures, but they do not establish whether an internalized graph can be repeatedly executed as a graph-specific parametric program without graph access at inference time. Therefore, we investigate executable parametric graph memory through a shared recurrent executor and graph-specific LoRA.

## Gap Matrix

| 方向 | 已能解决 | 仍不能回答 | RecurrentGRIP 的增量 |
|---|---|---|---|
| GRIP | 将图写入 LoRA；闭卷回答图问题 | 存的是事实还是程序；如何逐跳执行 | 显式分离 storage 与 execution |
| 更多 reasoning QA | 改善训练分布 | 是否形成可复用图算法 | 相同数据量下引入共享递归执行 |
| Depth-recurrent LLM | 增加测试时计算；潜在长度外推 | 重复执行的对象不是图专属记忆 | 把图 LoRA 作为被执行的参数算子 |
| 显式图神经网络 | 执行消息传递 | 推理阶段依赖输入图 | 将图编译进参数后闭卷执行 |
| KG-RAG | 路径可解释、更新方便 | 需要外部检索和图访问 | 研究无检索时的能力与成本边界 |
| Knowledge Editing | 局部事实写入或更新 | 多跳组合和算法执行 | 将多事实图记忆组织成可执行过程 |
| Adapter routing/composition | 选择或混合参数模块 | 不保证逐跳图状态 | 用 recurrence 产生有序执行轨迹 |

## 不是 Research Gap 的内容

以下表述不足以独立支撑论文：

- GRIP 的准确率还可以进一步提高；
- 使用新的 loss；
- 增加更多训练样本；
- 用动态权重混合不同 QA；
- 使用更大的 LoRA rank；
- 用 recurrent block 替换普通 block。

真正的 gap 必须与可验证发现绑定：

> 参数图记忆能否产生与 graph-hop 对齐、可因果操纵并能够长度外推的递归轨迹？

## 最近邻风险

### 风险 1：被归类为 Depth-Recurrent Transformer 应用

必须强调并验证：

- 图在推理阶段不可见；
- 图专属 LoRA 是 recurrence 操作的对象；
- 同一个共享执行器能切换不同图 adapter；
- adapter swap 会系统性改变递归轨迹。

### 风险 2：被归类为更多测试时计算

必须加入：

- FLOPs-matched baseline；
- unshared-depth baseline；
- repeated generation baseline；
- 同训练 token baseline。

### 风险 3：被归类为隐式 CoT

必须使用 hidden-state frontier probing、step deletion、step swapping 和 activation patching，证明中间状态具有图算法含义。

### 风险 4：被认为只在合成图有效

CLEGR 负责机制识别，Scene Graph 和四个真实 KG 负责外部有效性。

## Novelty Claims

### Claim N1

首次系统研究“内化图的可执行性”，而不只研究图能否写入参数。

### Claim N2

提出共享递归执行器与图专属 LoRA 的 storage–execution 分解。

### Claim N3

在无图输入条件下，将测试 recurrence 深度作为图推理计算深度，并研究未见 hop 外推。

### Claim N4

通过 frontier decoding 与 activation patching 建立参数图执行的机制证据。

## 当前 Novelty 判断

- 方法组合新颖性：中高；
- 机制问题新颖性：高于单纯结构改造；
- 主要风险：近期 recurrent reasoning 与参数图 memory 工作快速增加；
- 建议定位：机制 + 方法，而不是单纯 SOTA 模块论文。

# Literature Map for RecurrentGRIP

更新日期：2026-08-28

## 1. In-Parameter Graph Memory

### GRIP

- 作用：直接基础方法；将图生成的任务写入图专属参数，并在无图输入条件下推理。
- 已解决：图知识可以通过参数高效内化。
- 未解决：图参数是否形成逐步可执行的图算子；缺少显式 recurrence 和中间机制证据。
- 官方页面：https://arxiv.org/abs/2511.07457

### Parameter Memory vs. Retrieval: A Mechanistic Study of the Storage–Retrieval Gap in Graph Reasoning

- 作用：最新直接近邻；强调参数中存储图知识与查询时正确选择、组合参数记忆之间存在差距。
- 对本项目的约束：不能只做 adapter router；必须证明参数图的逐步执行机制。
- 官方页面：https://arxiv.org/abs/2608.25489

## 2. Knowing–Using Gap

### Towards Mechanistically Understanding the Knowing–Using Gap in LLM Finetuning

- 作用：支持“记住知识不等于能在推理中使用知识”。
- 关键启发：需要因果干预，而不能只观察最终准确率。
- 本项目迁移：在 recurrence 轨迹上做 activation patching，定位从存储到执行失败的步骤。
- 官方页面：https://arxiv.org/abs/2607.08393

## 3. Recurrent and Adaptive Computation

### Depth-Recurrent Transformer

- 作用：证明共享深度模块可以通过重复计算增加测试时推理深度。
- 与本项目差异：它研究通用语言推理，不负责把一张图编译进图专属 LoRA。
- 官方页面：https://openreview.net/forum?id=Q7boRE3kBy

### Recurrent Transformers Learn to Reason

- 作用：研究共享递归 Transformer 的推理和长度泛化能力。
- 与本项目差异：没有“共享执行机制 + 图专属参数存储 + 无图输入执行”的分解。
- 官方页面：https://openreview.net/forum?id=Au7WqYeoHb

## 4. Recurrent Graph Reasoning

### RANK / Recurrent Graph Reachability

- 作用：表明 recurrent computation 对图可达性和长度外推具有潜力。
- 与本项目差异：主要在显式图输入条件下执行图任务，不研究参数化存储的图。
- 官方页面：https://arxiv.org/abs/2608.11055

### Recurrent Attention for Graph Reasoning

- 作用：使用递归注意力进行图推理。
- 与本项目差异：图在推理阶段仍作为显式输入或结构表示；RecurrentGRIP 推理时不接触原图。
- 官方页面：https://arxiv.org/abs/2602.04445

## 5. Neural Algorithmic Reasoning

该方向提供三条重要方法论：

1. 用可控算法任务验证神经模型是否学习了程序；
2. 用中间状态与真实算法状态对齐；
3. 重点测试长度、规模和结构 OOD，而不只测试 IID 准确率。

RecurrentGRIP 借鉴的是评价逻辑，而不是把现成图网络直接接到 GRIP 上。

## 6. 相关工作分类框架

| 类别 | 存图位置 | 推理时有图 | 是否递归 | 是否有中间机制证据 | 与本项目关系 |
|---|---|---:|---:|---:|---|
| GRIP | LoRA 参数 | 否 | 否 | 有限 | 直接基础方法 |
| KG-RAG / GraphRAG | 外部图 | 是 | 可选 | 路径可见 | 强对照 |
| Recurrent Transformer | 共享模型参数 | 否 | 是 | 部分 | 执行机制来源 |
| Neural Graph Reasoning | 模型 + 输入图 | 是 | 可选 | 常见 | 算法评测来源 |
| Knowledge Editing | 模型参数 | 否 | 通常否 | 局部编辑指标 | 辅助近邻 |
| RecurrentGRIP | 图专属 LoRA + 共享执行器 | 否 | 是 | frontier + patching | 目标方法 |

## 7. 文献核验规则

- 所有论文标题、作者、年份和 venue 在写入论文正文前必须再次从 arXiv/OpenReview 官方页面核验。
- 当前文件只记录与方法设计直接相关的近邻，不把搜索结果数量等同于 novelty 证明。
- 2026 年 8 月之后出现的新论文必须在投稿前重新查新。

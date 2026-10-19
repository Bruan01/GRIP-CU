# Literature Map: Candidate Relation Learning

更新日期：2026-10-19

| 主题 | 对当前主线的作用 |
|---|---|
| GRIP | 提供 graph-specific LoRA、graph-context/QA 训练和候选关系学习位置 |
| InfoNCE / contrastive KGC | 提供候选决策集上的对比目标 |
| Hard-negative mining | 提供从模型错误或高分违例构造困难负样本的动机 |
| KG false-negative filtering | 提醒候选缺失不等于事实为假，必须使用 train-only known-fact audit |
| Error memory | 当前实验的局部设计：冻结 B1 的 gold-conditioned 混淆统计 |

## 研究边界

已有工作覆盖通用 hard negative、model-generated negative、adaptive hardness 和 contrastive KGC。因此当前可检验的局部主张是：在相同 Original GRIP 预算下，train-only ErrorMemory-9 是否比 Random-9 更适合候选关系学习。任何提升必须由 NELL23K 和 FB15K237-2 实验验证。

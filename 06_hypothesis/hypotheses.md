# Hypotheses: ErrorMemory-9 + InfoNCE

更新日期：2026-10-19

## H1：边界假设

冻结 B1 的 gold-conditioned 错误与 score violation，比 Random-9 更接近候选关系决策边界。

**预测：** ErrorMemory-9 的 closed-set relation MRR/Hits 优于 Random-9。

## H2：生成保持假设

候选级 InfoNCE 不应破坏 Original GRIP 的自由生成能力。

**预测：** Random-9 和 ErrorMemory-9 的 free-generation EM 不显著低于 B1。

## H3：数据集稳健性假设

如果方法利用的是一般的候选决策信号，而不是 NELL23K 特有的标签格式，它应在 FB15K237-2 上保留方向一致的收益。

**预测：** 两个数据集的候选指标方向一致；若不一致，按数据集分别解释。

## 证伪条件

- ErrorMemory-9 不优于 Random-9；
- 候选排序提升但 free-generation EM 明显下降；
- 训练/验证/测试泄漏或候选 ID 不一致；
- 结果只能由不同训练步数、adapter 或模型缓存解释。

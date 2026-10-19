# Claim Boundaries

更新日期：2026-10-19

## 可以声称

- 我们在 Original GRIP 的候选关系学习阶段测试了 Random-9 和 ErrorMemory-9；
- ErrorMemory-9 使用冻结 B1 的 train-only 错误、候选分数违例和固定 K=9 候选池；
- 结果可以支持或否证“模型条件 hard negatives 改善候选关系辨别”的局部假设。

## 不能提前声称

- ErrorMemory-9 已经提升性能；
- 方法解决 OOV 或开放世界 KG 的真假判断；
- 方法提出了通用 KG 负采样理论；
- 候选指标提升必然转化为自由生成提升；
- NELL23K 或 FB15K237-2 的单次结果代表所有 KGQA 数据集。

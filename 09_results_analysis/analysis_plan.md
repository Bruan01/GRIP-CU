# Analysis Plan

更新日期：2026-10-19

## 首要比较

对 NELL23K 和 FB15K237-2 分别比较：

1. Original GRIP/B1；
2. Random-9 + InfoNCE；
3. ErrorMemory-9 + InfoNCE。

## 分析问题

- ErrorMemory-9 是否改善 candidate MRR/Hits？
- free-generation EM 是否保持？
- 改善是否集中在 B1 曾经混淆的关系？
- 候选有效率、已知事实过滤和 OOV 是否解释结果？
- 两个数据集是否出现一致或相反方向？
- 是否存在训练预算、adapter、seed 或 task provenance 混淆？

## 输出

- arm × dataset 主结果表；
- candidate score gap 和 MRR/Hits 图；
- B1 error-memory 覆盖率与边界切片；
- free-generation EM/OOV/wrong-in-list 分析；
- 失败样本和停止条件记录。

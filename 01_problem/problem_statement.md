# Problem Statement: Candidate Relation Learning in GRIP

更新日期：2026-10-19

## 1. 研究对象

GRIP 将图信息写入图专属 LoRA，并在推理时通过问题和参数化图记忆回答关系问题。当前实验不改变图存储机制，而聚焦候选关系学习阶段：模型面对一个 gold relation 和若干候选关系时，负样本是否足够接近真实决策边界。

## 2. 基线问题

Original GRIP/B1 的候选负样本没有利用模型自身暴露出的训练错误结构：

- 完全随机的候选可能远离决策边界；
- 只按当前分数选 hard negative 可能放大偶然误差；
- 已知事实没有被稳定地从负样本池中剔除。

## 3. 当前问题

> 在相同的 Original GRIP 生成目标和 Stage 2 预算下，使用冻结 B1 的训练集错误记忆与候选分数违例构造的 ErrorMemory-9，是否比 compute-matched Random-9 更能改善候选关系辨别，同时不损害自由生成？

## 4. 比较路径

1. Original GRIP/B1；
2. Random-9 + InfoNCE；
3. ErrorMemory-9 + InfoNCE。

## 5. 证据范围

首轮在 NELL23K 和 FB15K237-2 上运行。主指标包括 free-generation EM、closed-set relation MRR/Hits、wrong-in-list、OOV、候选有效率和 train-only provenance。该实验不研究递归执行器、不声称解决 OOV，也不提出通用 KG 负采样理论。

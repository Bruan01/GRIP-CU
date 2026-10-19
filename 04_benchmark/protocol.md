# Candidate-learning evaluation protocol

更新日期：2026-10-19

## 1. 固定比较

每个数据集必须运行以下三条路径：

| Arm | 目标 |
|---|---|
| Original GRIP/B1 | generation-only 基线 |
| Random-9 + InfoNCE | 检验 InfoNCE 本身的收益 |
| ErrorMemory-9 + InfoNCE | 检验 B1-conditioned boundary negatives 的额外收益 |

三条路径使用同一基座模型、Stage 1 adapter、task JSON、Stage 2 epoch、batch、gradient accumulation、seed 和评估文件。

## 2. 候选集合

对每个 relation QA，候选关系来自训练 split 的关系词表：

```text
P(q) = R_train \ {gold(q) ∪ known_train_relations(head(q), tail(q))}
```

正式候选数固定为 `K=9` 个负样本。Random-9 从合法训练关系池均匀采样；ErrorMemory-9 使用冻结的 B1 错误记忆、B1 score violation 和均匀概率下限采样。

## 3. 数据泄漏控制

- ErrorMemory 只读取 `train.txt` 的 QA、关系词表和已知事实；
- `valid.txt`、`test.txt` 不参与候选构造；
- B1 adapter 在候选采样前冻结；
- sampler bank、score dump、rollout manifest 的 question ID 必须一致；
- 正式训练使用完整 task JSON，不使用 smoke 子集；
- FB15K237-2 的 raw relation ID 必须先通过 `rel2text.json` 映射到 task 关系文本。

## 4. 报告指标

每个 arm 和数据集至少报告：

- free-generation Exact Match；
- closed-set relation MRR、Hits@1、Hits@3、Hits@10；
- wrong-in-list、OOV 和边界错误切片；
- 候选有效率、已知事实过滤数和候选覆盖率；
- 训练步数、候选 loss、generation loss、显存和延迟。

正式主结果使用多个 seed；单 seed 只能作为 smoke 或方向性证据。

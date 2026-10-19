# Baseline Registry

更新日期：2026-10-19

当前只保留与候选关系学习主线直接相关的三条路径：

| ID | Arm | 目的 | 候选关系学习 |
|---|---|---|---|
| B1 | Original GRIP/B1 | generation-only 基线 | 无 InfoNCE |
| B2 | Random-9 + InfoNCE | 测量随机负样本和 InfoNCE 的效果 | 训练关系词表均匀采样 9 个负样本 |
| B3 | ErrorMemory-9 + InfoNCE | 测量 B1-conditioned boundary negatives 的额外效果 | 冻结 train-only ErrorMemory/violation bank |

三条路径共享 Original GRIP 代码、Stage 1 adapter、基座模型、task JSON、Stage 2 预算、seed 和评估协议。差异只允许出现在 Stage 2 候选负样本和对应 InfoNCE 项。

## 数据集

- NELL23K；
- FB15K237-2。

两套数据位于本地未跟踪目录 `04_benchmark/data/`。FB15K237-2 的 raw relation ID 必须使用 `rel2text.json` 映射到 task 标签。

## 公平性和证据要求

- B1/B2/B3 使用相同生成目标和完整 task budget；
- ErrorMemory、候选 score 和已知事实过滤只来自 train split；
- valid/test 只用于评估；
- 记录 branch HEAD、baseline 子模块 commit、模型/adapter/data hash、GPU、seed 和训练步数；
- 未通过 wiring 审计前不启动正式训练；
- 没有结果前不把 B3 写成已经提升的主方法。

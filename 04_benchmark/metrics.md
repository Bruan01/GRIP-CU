# Metrics

更新日期：2026-10-19

## 自由生成

- Exact Match；
- wrong-in-list rate；
- OOV rate；
- 按关系频次、候选是否包含 B1 错误和实体对切片的 EM。

## 闭集候选关系

- MRR；
- Hits@1、Hits@3、Hits@10；
- gold-vs-negative score gap；
- pairwise confusion probability；
- valid-negative rate 和候选覆盖率。

## 训练与资源

- generation loss；
- candidate InfoNCE loss；
- 候选和生成 loss 的分阶段曲线；
- 完整训练步数、有效 QA 数和 sampler K；
- trainable parameters、adapter size、训练时间、推理延迟和 peak GPU memory。

## 统计

Pilot 可使用单 seed 做方向判断；正式主结果至少使用 3 个 seed，报告均值、标准差、置信区间和配对效果量。没有完整 provenance 时不合并不同数据集或不同 adapter 的结果。

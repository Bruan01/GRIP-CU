# 08 Experiments

当前实验区只维护原始 GRIP 候选关系学习的三臂比较：

1. Original GRIP/B1；
2. Random-9 + InfoNCE；
3. ErrorMemory-9 + InfoNCE。

## 当前方法目录

- `Hard_Negative_Contrastive_GRIP/`：ErrorMemory-9、Random-9、离线 B1 rollout/score、wiring 审计和 Stage 2 训练入口；
- `13_base_method/grip-exp/`：未改动的 Original GRIP/B1 基线代码。

## 数据集

首轮使用 `NELL23K` 和 `FB15K237-2`。完整数据、processed graph record 和 task JSON 位于本地未跟踪目录 `04_benchmark/data/`。

## 实验顺序

```text
preflight
  -> frozen B1 train rollout
  -> frozen B1 full-vocabulary score
  -> ErrorMemory-9 freeze
  -> full-task wiring audit
  -> Original GRIP/B1 vs Random-9 vs ErrorMemory-9
  -> paired evaluation and analysis
```

每次实验记录 branch HEAD、baseline 子模块 commit、数据/adapter hash、seed、GPU、预算、日志和结论。失败实验同样保留。

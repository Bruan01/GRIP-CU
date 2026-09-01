# PriorityDistillGRIP

当前候选主线：把图上的关键路径优先机制从“推理时检索组件”改造成“训练时教师”。模型训练时可读取路径，测试时只保留普通 LoRA 参数，不访问图、路径检索器或路由器。

## 版本

- [`v0_1_oracle_priority_smoke_2026-09-01/`](v0_1_oracle_priority_smoke_2026-09-01/)：NELL23K 严格 1–4 hop oracle upper-bound gate。先判断完美路径选择是否值得蒸馏，再决定是否开发 learned prioritizer。

## 不可跨越的决策边界

1. v0.1 未通过前，不实现 PathMind 式语义 scorer、DPO 或检索器。
2. `GO_LEARNED_PRIORITIZER` 需要三个 seeds，并同时超过 answer-only、equal-token More-QA、random path 和 all paths。
3. 测试阶段不得读取图或候选路径。
4. Original GRIP 位于 `13_base_method/grip-exp/`，保持只读；每个实验版本保留独立快照。

# PriorityDistillGRIP

当前候选主线：把图上的关键路径优先机制从“推理时检索组件”改造成“训练时教师”。模型训练时可读取路径，测试时只保留普通 LoRA 参数，不访问图、路径检索器或路由器。

## 版本

- [`v0_1_oracle_priority_smoke_2026-09-01/`](v0_1_oracle_priority_smoke_2026-09-01/)：NELL23K 严格 1–4 hop oracle upper-bound gate。先判断完美路径选择是否值得蒸馏，再决定是否开发 learned prioritizer。
- [`v0_1_1_oracle_stage2_diagnostic_2026-09-01/`](v0_1_1_oracle_stage2_diagnostic_2026-09-01/)：只运行 oracle 的 Stage-2 延长诊断，保存每阶段 checkpoint，并同时评估 graph-free 与 oracle-evidence。用于区分训练不足和路径末节点复制捷径。
- [`v0_1_2_controlled_protocol_2026-09-01/`](v0_1_2_controlled_protocol_2026-09-01/)：固定 token 预算公平比较 direct answer-only 与 oracle two-stage。
- [`v0_1_3_candidate_selection_anti_copy_2026-09-01/`](v0_1_3_candidate_selection_anti_copy_2026-09-01/)：候选路径选择、隐藏终点 anti-copy、Stage-2 replay 诊断。
- [`v0_1_4_explicit_path_selection_2026-09-01/`](v0_1_4_explicit_path_selection_2026-09-01/)：显式 path index 选择与 anti-copy 协议。
- [`v0_1_5_bridge_supervision_2026-09-01/`](v0_1_5_bridge_supervision_2026-09-01/)：bridge supervision / direct answer-only 公平基线。
- [`v0_1_6_joint_trace_answer_2026-09-02/`](v0_1_6_joint_trace_answer_2026-09-02/)：graph-free trace + answer、纠正生成预算、候选集解码与 seed 43/44 公平 follow-up；当前未证明 joint 稳定优于 direct。

## 不可跨越的决策边界

1. v0.1 未通过前，不实现 PathMind 式语义 scorer、DPO 或检索器。
2. `GO_LEARNED_PRIORITIZER` 需要三个 seeds，并同时超过 answer-only、equal-token More-QA、random path 和 all paths。
3. 测试阶段不得读取图或候选路径。
4. Original GRIP 位于 `13_base_method/grip-exp/`，保持只读；每个实验版本保留独立快照。

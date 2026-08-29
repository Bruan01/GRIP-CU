# CLEGR Two-Hour Pilot Plan

更新日期：2026-08-28

## 1. Pilot 问题

在只使用 1–2 hop 任务训练后，测试时增加共享 recurrence，能否改善未见的 3–4 hop 闭卷图推理？

## 2. 时间预算

单次 pilot 的 GPU 时间上限：2 小时。超过 3 小时强制停止并保留部分结果。

## 3. 数据

### v1 可证伪子任务

CLEGR-Reasoning 同时包含 Filter、Aggregation、Topology 和 PathReasoning，不能把所有 reasoning 问题直接解释为固定 hop。首轮 Pilot 因此只使用 `StationShortestCount`：从问题恢复两个端点，并在 `edge_index` 上重新计算真实无向最短距离。数据标签必须等于 `max(该距离 - 1, 0)`，否则拒绝样本。其他 PathReasoning 类型留到 v1.1。

- Dataset：CLEGR；
- 图数量：先使用 8–16 张小图；
- 节点数：20–50；
- 训练：1–2 hop；
- 验证：1–2 hop；
- 测试：3–4 hop；
- 节点名称：随机无语义标识；
- 每条样本保存 shortest distance、relation path 和 frontiers；
- 自动排除一跳或更短路径捷径。

## 4. 模型

- 优先 Qwen2.5-0.5B 或 1.5B；
- LoRA rank：4 或 8；
- recurrent executor：1 个共享 block 或连续 2 个共享 block；
- recurrence sweep：\(K=1,2,3,4,5\)；
- Pilot 使用固定深度，不实现 halt head；
- 训练 token 和 QA 数对所有方法保持一致。

## 5. 对照

### P0：Base LM

不加载图 adapter，检查泄漏。

### P1：Original GRIP

原始一次前向推理。

### P2：GRIP + More Reasoning QA

与 RecurrentGRIP 使用完全相同的训练问题，不使用 recurrence。

### P3：Fixed-Depth RecurrentGRIP

同一个 executor 重复 \(K\) 次。

## 6. 记录内容

每条问题记录：

- graph ID；
- question ID；
- true hop；
- recurrence \(K\)；
- raw response；
- parsed answer；
- correctness；
- latency；
- peak memory；
- 每步 pooled hidden state；
- 使用的 adapter ID。

## 7. 成功条件

至少满足两项：

1. 3–4 hop 比 Original GRIP 提升至少 5 个绝对百分点；
2. 3–4 hop 上 \(K=3/4\) 明显优于 \(K=1\)；
3. 最优 \(K\) 与 true hop 正相关；
4. 1–2 hop 性能下降不超过 2 个百分点；
5. adapter shuffle 明显破坏性能。

## 8. 失败诊断顺序

若 pilot 失败，按以下顺序检查：

1. 原始 GRIP 是否成功记住 1-hop 事实；
2. 多跳数据是否存在捷径或答案解析错误；
3. recurrent block 是否真正共享权重；
4. graph LoRA 是否实际作用于 recurrent block；
5. hidden state 是否出现数值发散或表示坍缩；
6. 增加 recurrence 是否只是造成 overthinking；
7. 执行器是否需要跨图 meta-training。

## 9. Pilot 后决策

- 通过：进入 v2 动态停止和机制分析；
- 部分通过：只调整 executor 位置、残差门和训练深度，不扩展六数据集；
- 未通过：停止方法扩张，形成负结果分析，并比较 SuccessorGRIP 备选方向。

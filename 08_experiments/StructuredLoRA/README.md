# StructuredLoRA

StructuredLoRA 是 GRIP-CU 当前候选主线，目标是在**相同总 LoRA rank、相同训练数据和相同推理预算**下，将参数化图知识组织为有序的深度残差子空间，而不是把不同结构复杂度的知识压缩到同一个无结构低秩矩阵。

## 当前方法假设

对每个 LoRA 目标层，将总 rank 拆成有序组：

```text
G1：原子事实和基础参数记忆
G2：二阶组合残差
G3：三阶组合残差
G4：更长或更复杂的支撑残差
```

深度为 `d` 的样本累计激活 `G1...Gd`，而不是从互相可交换的平坦专家中只选择一个。正式候选方法还包括：

1. ordinal/cumulative router：预测 `P(depth >= g)`；
2. group-wise orthogonal subspaces；
3. depth-local credit assignment：前向使用完整前缀，反向主要更新当前深度残差组。

## 研究边界

- NELL23K 标准关系预测问题不提供可信的显式 query hop；
- NELL23K 使用的是 train graph 上的 **support depth proxy**；
- 严格的 hop-specialization 因果主张由 exact-hop path QA 支撑；
- `拆 rank + router + orthogonality` 本身不足以构成创新，核心必须是有序前缀结构和深度局部信用分配；
- learned router 只有在 oracle-prefix 等预算实验获胜后才实现。

## 版本

| Version | Status | Purpose |
|---|---|---|
| [`v0_1_depth_data_audit_2026-08-31`](v0_1_depth_data_audit_2026-08-31/) | Complete / GO | 构建可信 support-depth 标签和 exact-hop 任务，决定是否进入 oracle-prefix pilot |
| `v0_2_oracle_prefix_smoke` | Next | 同 rank 比较 monolithic、static split、flat experts 与 ordered prefix |
| `v0_3_learned_ordinal_router` | Blocked by v0.2 | 只有 oracle-prefix 通过后才训练预测路由 |
| `v0_4_nell23k_official_pilot` | Planned | 1.5B 筛选后运行 7B 正式配置和三随机种子 |

## 当前决策

v0.1 在完整 NELL23K split 上得到：

- `34,216` 条 support-depth 标签；
- `1,024` 条严格 1/2/3/4-hop path QA；
- test 中 depth 2/3/4 分别有 `670/2260/427` 条无向 support-path 样本；
- relation-depth NMI 为 `0.159158`，depth 不是 relation ID 的简单复制；
- 旧诊断预测 `768/768` 成功重新连接到新 depth 标签。

决策：**进入 v0.2 oracle-prefix equal-rank smoke，但不直接实现 learned router。**

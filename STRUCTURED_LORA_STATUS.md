# StructuredLoRA Research Status

更新日期：2026-08-31

## 当前主线

**Depth-Structured LoRA for In-Parameter Graph Reasoning**

研究问题：GRIP 将不同结构支撑深度、不同关系频率和不同组合复杂度的图知识压缩进同一个无结构 LoRA 子空间，是否产生跨深度参数干扰？在相同总 rank 下，将 LoRA 组织为有序累计深度残差，是否能提升闭卷图关系预测和可控多跳推理？

## 当前候选方法

```text
G1：基础事实记忆
G2：二阶残差
G3：三阶残差
G4：长路径残差
```

深度 `d` 累计激活 `G1...Gd`。核心候选贡献是 ordered prefix、ordinal routing 和 depth-local credit assignment，不是普通的 rank split 或 MoE-LoRA。

## 已完成

- [x] 近邻工作边界：普通 rank experts/router/orthogonality 不足以支撑新颖性
- [x] NELL23K leave-one-edge-out support-depth 实现
- [x] Directed/undirected depth 与 censored bucket 导出
- [x] 严格 1/2/3/4-hop path QA 生成器
- [x] 34,216 条完整标签和 1,024 条 exact-hop QA
- [x] 旧诊断预测 768/768 重新连接新标签
- [x] 8 个纯 Python 测试与确定性 artifact 检查
- [x] v0.1 Go gate：GO_ORACLE_PREFIX

## 下一步

- [ ] 创建 `v0_2_oracle_prefix_smoke_2026-09-01`
- [ ] 实现 equal-rank monolithic、static split、flat oracle expert、ordered oracle prefix
- [ ] 实现 depth-local credit 和 permuted/random controls
- [ ] 在 Qwen2.5-0.5B + RTX 3090 上运行 exact-hop smoke
- [ ] oracle-prefix 通过后才创建 learned ordinal router

## 关键入口

1. [`08_experiments/StructuredLoRA/README.md`](08_experiments/StructuredLoRA/README.md)
2. [`08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/README.md`](08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/README.md)
3. [`08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/PROJECT_MEMORY.md`](08_experiments/StructuredLoRA/v0_1_depth_data_audit_2026-08-31/PROJECT_MEMORY.md)
4. [`TODO_STRUCTURED_LORA.md`](TODO_STRUCTURED_LORA.md)

## Stop rule

如果 perfect/oracle routing 下的 ordered prefix 不能超过同总 rank monolithic LoRA，则终止该方向，不继续实现 learned router，也不使用增加 rank 掩盖失败。

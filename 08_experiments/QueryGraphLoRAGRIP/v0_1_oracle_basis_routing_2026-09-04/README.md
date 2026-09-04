# QueryGraph-LoRA GRIP v0.1 — Oracle-First Basis Routing

- **实验编号**：E09
- **日期**：2026-09-04
- **状态**：`SUPERSEDED_BY_V0_2_AFTER_E03_STOP`
- **主部件**：L4.03 参数分组与路由
- **来源机制**：iLoRA（arXiv:2605.30179）的样本级潜在图与图条件 LoRA
- **当前边界**：E03 已完成 validation-only 机制边界审计并停止 decoder-primary；本目录保留为设计注册，v0.2 是实际本地实现版本。

## 一句话问题

在推理时不访问 KG/subgraph 的前提下，仅由 question tokens / hidden states 推断的潜在结构，能否动态选择 GRIP 的参数记忆子空间，缓解关系、跳数与组合异质性造成的静态 LoRA 干扰？

## 方法差异

iLoRA 从显式 multi-entity input 推断样本图，并由 hypernetwork 生成输入特定 LoRA A。GRIP 不能直接照搬该输入假设。本实验改为：

```text
question only -> latent query structure/router -> shared LoRA basis mixture -> answer
ΔW(x) = Σ_m π_m(q) · B_m A_m
```

所有 query-specific 路由信号只来自问题文本或基础模型隐藏状态；validation/test 不访问 KG、候选子图、gold path 或 gold relation label。

## 执行顺序

1. 等待 E03 validation-only gate；
2. Phase A：oracle basis routing 上界；
3. oracle 过门后，Phase B：deterministic query router；
4. learned router 过门后，Phase C：Bayesian latent query graph 与 uncertainty；
5. 最后才回到官方 Qwen2.5-7B、三 seeds 主实验。

详见 `EXPERIMENT_PLAN.md` 与 `configs/experiment_registry.json`。

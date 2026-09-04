# Project Memory — QueryGraph-LoRA GRIP v0.1

## Frozen decisions (2026-09-04)

1. E09 是条件候选，E03 仍为 active experiment。
2. v0.1 已被 E03 STOP 结果推进的 v0.2 本地实现取代；本目录只保留设计注册，不作为服务器运行入口。
3. Original GRIP `13_base_method/grip-exp/` 只读。
4. 推理时只从 question tokens/hidden states 路由，禁止 query-specific graph、gold path/relation/hop 与答案候选。
5. 首先运行 oracle basis-routing upper bound；oracle 失败即停止 learned/Bayesian 模块。
6. 参数、token、steps、decoder、seeds 必须公平匹配。
7. 推荐共享 basis mixture，而不是逐样本生成每层完整 LoRA 权重。
8. iLoRA 是最近机制来源，但其 microbiome/Molweni 结果不构成 KGQA 有效性证据。

## Immediate next action

等待 E03 `PRELIMINARY_GO_D2` 或后续明确的机制决策；只有确认 decoder 不是主要瓶颈且 L4 参数读取仍有空间，才执行 Phase-A oracle routing。

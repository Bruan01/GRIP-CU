# Experiment Matrix

更新日期：2026-10-19

| 阶段 | Original GRIP/B1 | Random-9 + InfoNCE | ErrorMemory-9 + InfoNCE |
|---|---|---|---|
| NELL23K read-only preflight | 同一完整 task 契约 | 同一预算、随机训练关系池 | 同一预算、冻结 train-only bank |
| NELL23K 主实验 | generation-only | generation + candidate InfoNCE | generation + ErrorMemory candidate InfoNCE |
| FB15K237-2 read-only preflight | 同一完整 task 契约 | 同一预算、随机训练关系池 | 同一预算、冻结 train-only bank |
| FB15K237-2 主实验 | generation-only | generation + candidate InfoNCE | generation + ErrorMemory candidate InfoNCE |

## 固定变量

- base model、LoRA target modules、rank/alpha；
- Stage 1 adapter、Stage 2 epoch、batch、gradient accumulation；
- seed、task file、评估文件和候选数 `K=9`；
- 只使用训练 split 构造关系词表、错误记忆、候选分数和已知事实过滤。

## 主指标

- free-generation EM；
- closed-set relation MRR/Hits；
- wrong-in-list、OOV、边界错误切片；
- generation loss、candidate InfoNCE loss、有效候选率和资源开销。

没有通过数据 provenance、question ID、K=9 和 train-only wiring 审计前，不启动正式训练。

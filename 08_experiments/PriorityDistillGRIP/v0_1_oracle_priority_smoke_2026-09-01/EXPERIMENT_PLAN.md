# Experiment Plan

- **Problem**: 训练期关键路径监督能否提高 graph-free 参数知识内化？
- **Method thesis**: 路径优先选择作为训练教师，比更多 QA、随机路径或未筛选路径提供更高质量的参数更新。
- **Date**: 2026-09-01

## Claim Map

| Claim | Minimum convincing evidence | Block |
|---|---|---|
| C1 路径选择质量有效 | oracle 在等 token 下超过 More-QA、random、all-paths | B1/B2 |
| C2 收益被内化到参数 | validation/test 无图访问仍提升 | B1 |
| Anti-claim 不是更多预算 | 实际 input/supervised/padded tokens 与 steps 全部报告 | B2 |

## Blocks

### B0 — Pipeline sanity
- 数据：NELL23K strict exact-hop。
- 检查：split SHA、candidate leakage、gold position、graph-free eval prompt。
- 状态：已通过。

### B1 — One-seed oracle gate
- 系统：五个注册方法，seed 42。
- 指标：test accuracy、macro-hop、worst-hop、d1–d4、deep 3/4。
- 成功：全部注册 gate 暂时通过，输出 `PRELIMINARY_GO`。
- 失败：`PRELIMINARY_STOP`，不开发 learned prioritizer。

### B2 — Budget audit
- 指标：Stage-1 input tokens、supervised tokens、padding tokens、optimizer steps、wall time、peak GPU memory、prompt truncation rate。
- 成功：四个 Stage-1 方法 input-token 相对差距 ≤ 1%，且 prompt 截断率均为 0。
- 失败：先修预算或上下文长度，不解释准确率。

### B3 — Three-seed confirmation
- 前置：B1 preliminary go。
- Seeds：42/43/44。
- 成功：最终 `GO_LEARNED_PRIORITIZER`。
- 失败：`STOP_PRIORITY_DISTILL`。

### B4 — Cross-dataset replication
- 前置：B3 final go。
- 数据：优先 FB15K-237 exact-hop 审计版。
- 目的：排除 NELL23K 特例。
- 本版本不执行。

## Run Order

| Milestone | Run | Gate | Priority |
|---|---|---|---|
| M0 | static + supervision audit | READY_FOR_WSL_GPU | DONE |
| M1 | five methods × seed 42 | PRELIMINARY_GO/STOP | MUST |
| M2 | inspect budget and errors | no confound | MUST |
| M3 | five methods × seeds 42/43/44 | final Go/Stop | CONDITIONAL |
| M4 | learned prioritizer v0.2 | only after final Go | CONDITIONAL |

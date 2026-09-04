# QueryGraph-LoRA GRIP v0.2 — Oracle Basis Routing

- E09 Phase-A；状态：`LOCAL_IMPLEMENTATION_READY_FOR_WSL_VALIDATION`
- NELL23K exact-hop、Qwen2.5-0.5B、validation-only、test locked。
- 问题：等参数预算下，hop/path oracle 路由 4×rank-2 LoRA basis 是否优于静态 rank-8？

这是证伪型 upper-bound。只有两 seeds 的 oracle 同时通过 overall、novel、seen 保持和 control separation gate，才开发 question-only learned router；否则停止该主线。

服务器按 `REMOTE_RUNBOOK.md` 执行；可直接发送 `NEXT_STEP_WSL_PROMPT.md` 给服务器 Codex。

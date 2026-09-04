# 给服务器 Codex 的 Prompt

只执行和审计，不在服务器开发/修改代码：
1. 在 `/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU` 拉取分支 `wsl/nell23k-smoke-20260829`，报告 HEAD/status，不 reset/clean。
2. 逐字执行 `08_experiments/QueryGraphLoRAGRIP/v0_2_oracle_basis_routing_2026-09-04/REMOTE_RUNBOOK.md`。
3. Preflight 失败就停止回传日志。
4. 先对服务器 E03 原始 run 执行复算审计。
5. E09 只跑 B1/B2/B4/O1/O3 × seeds 43/44，validation-only；不得打开 test。
6. gate STOP 立即结束；不开发 learned/Bayesian router，不上 7B。
7. 生成 SHA256 manifest，回传环境、配置 hash、公平性计数、REPORT、suite、所有 run_summary、异常和 test-access audit。
8. 不修改 `13_base_method/grip-exp/`，不提交代码。

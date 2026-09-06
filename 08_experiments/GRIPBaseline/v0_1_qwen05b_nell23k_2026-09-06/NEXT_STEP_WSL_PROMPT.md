# Prompt for the server Codex

```text
请在当前 WSL2 + RTX 3090 服务器上执行，不要修改任何源代码。先进入
/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU，拉取
origin/wsl/nell23k-smoke-20260829 的最新提交，并确认 git rev-parse HEAD
与本实验新增代码的 commit 一致。保持仓库中与本实验无关的未跟踪文件不动。

阅读以下文件：
- 08_experiments/GRIPBaseline/v0_1_qwen05b_nell23k_2026-09-06/README.md
- 08_experiments/GRIPBaseline/v0_1_qwen05b_nell23k_2026-09-06/EXPERIMENT_PLAN.md
- 08_experiments/GRIPBaseline/v0_1_qwen05b_nell23k_2026-09-06/REMOTE_RUNBOOK.md

先执行 runbook 的 preflight，确认 qwen-0.5b mapping、Python/CUDA、
processed_test.json 和 Qwen2.5-0.5B snapshot。优先做 NUM_TEST=16 的 smoke，
只把它当作管线检查；成功后使用新的稳定 RUN_ID 执行完整 processed_test.json
正式基线。默认 task generator 为 qwen-7b，fine-tuning/inference backbone
为 Qwen2.5-0.5B，保持 no_graph_context=True、use_subgraph=False、
index_format=False。记录 task generator 与 backbone 分开、实际测试行数、
数据和模型 SHA256、完整命令、日志、task cache、adapter、predictions、
最终 em/f1/hit 以及任何 OOM/中断/恢复情况。

正式 run 如中断，只用同一个 RUN_ID 加 --resume-outputs 恢复，不要重新生成
一个不同协议的结果。完成后不要修改代码或提交服务器临时改动，只回传：
1) commit hash；2) smoke/full 的运行目录和日志；3) prediction 文件；
4) test/prediction 行数；5) em/f1/hit；6) 数据、模型、task cache 的 SHA256；
7) 运行异常和是否使用了 resume。
```

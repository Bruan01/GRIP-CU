# WSL Codex Prompt — Run StructuredLoRA v0.2

```text
先拉取 GRIP-CU 当前分支，然后阅读：

1. TODO_STRUCTURED_LORA.md
2. 08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/README.md
3. 08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/PROJECT_MEMORY.md
4. 08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/configs/oracle_prefix_smoke.json

不要修改 13_base_method/grip-exp，不要覆盖 v0.1 或已有 results。

在 WSL2 + RTX 3090 + guardenv 中执行：

conda activate guardenv
cd 08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01
RUN_ID=wsl3090_oracle_prefix_smoke_01 bash configs/run_wsl_smoke.sh

如果模型已经下载，设置：

MODEL_NAME_OR_PATH=/你的/Qwen2.5-0.5B-Instruct/目录

完成后检查：

results/runs/wsl3090_oracle_prefix_smoke_01/REPORT.md
results/runs/wsl3090_oracle_prefix_smoke_01/suite_summary.json
results/runs/wsl3090_oracle_prefix_smoke_01/suite_metrics.csv

必须审计：
- 八个方法是否全部完成；
- trainable parameter 数是否公平；
- ordered_prefix 相对 monolithic/flat/permuted/non_nested 的结果；
- d1/d2/d3/d4、macro-hop、worst-hop；
- gradient cosine；
- group knockout；
- 是否为 PRELIMINARY_GO 或 PRELIMINARY_STOP。

不要在 preliminary stop 后实现 router。若 preliminary go，再运行：

RUN_ID=wsl3090_oracle_prefix_full_01 bash configs/run_wsl_full_seeds.sh

将结果、报告和必要日志提交并推送当前远程分支。不要提交模型缓存；adapter_model.pt 若超过 GitHub 单文件限制，则保留其 SHA256 和本地路径，不强行提交。
```

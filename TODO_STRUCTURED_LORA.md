# StructuredLoRA 下一步 TODO

更新日期：2026-08-31

## 当前状态

v0.2 方法、对照、训练器、评估器和静态验证已经完成：

```text
08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/
```

当前唯一主任务是在 WSL2 + RTX 3090 上运行 oracle-prefix smoke，而不是继续改方法或实现 learned router。

完整 WSL 交接 Prompt：

- [`08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/NEXT_STEP_WSL_PROMPT.md`](08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01/NEXT_STEP_WSL_PROMPT.md)

## WSL 运行

```bash
git pull --ff-only
conda activate guardenv
cd 08_experiments/StructuredLoRA/v0_2_oracle_prefix_smoke_2026-09-01
RUN_ID=wsl3090_oracle_prefix_smoke_01 bash configs/run_wsl_smoke.sh
```

如使用本地模型：

```bash
MODEL_NAME_OR_PATH=/path/to/Qwen2.5-0.5B-Instruct \
RUN_ID=wsl3090_oracle_prefix_smoke_01 \
bash configs/run_wsl_smoke.sh
```

脚本支持断点续跑：已完成 method/seed 自动跳过，不完整目录自动重建。

## 已注册比较

```text
monolithic rank=8
static split 4x2 all-on
flat oracle expert 4x2
ordered oracle prefix 4x2
ordered prefix + depth-local credit
permuted depth labels
fixed random group order（symmetry control）
non-nested random masks（decisive structure control）
```

## Smoke 后

检查：

```text
results/runs/<RUN_ID>/REPORT.md
results/runs/<RUN_ID>/suite_summary.json
results/runs/<RUN_ID>/suite_metrics.csv
```

- `PRELIMINARY_STOP`：记录并停止，不实现 router；
- `PRELIMINARY_GO`：运行 `bash configs/run_wsl_full_seeds.sh`；
- 三 seeds 最终 `GO_LEARNED_ROUTER`：才进入 v0.3；
- 三 seeds `STOP_STRUCTURED_LORA`：终止该方向。

## 禁止事项

- 不修改 `13_base_method/grip-exp/`；
- 不覆盖 v0.1 或已有 run ID；
- 不实现 learned router；
- 不增加 total rank 掩盖失败；
- 不把 NELL23K support depth 写成 ground-truth query hop；
- 不把 `>4_or_unreachable` 写成全局不可达；
- 不把固定 group permutation 的结果当成有序性的决定性证据。

# E03 EntityConstrained-GRIP Phase-A 远程结果审计

macOS 没有服务器原始 predictions；本目录提供服务器侧复算工具，不把聊天数字当 raw evidence。

```bash
python 09_results_analysis/2026-09-04_entity_constrained_grip_phase_a_audit/scripts/audit_e03_results.py --run-dir 08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/results/runs/wsl3090_entity_decoder_validation_20260904_02
```

输出到 `<run-dir>/e03_audit/`：`E03_RESULT_AUDIT.{json,md}`、`metric_reproduction.json`、`artifact_manifest.json`。

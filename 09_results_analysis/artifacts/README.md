# Trackable Experiment Audit Artifacts

本目录只保存可以进入 Git 的小型、可审计实验产物，不保存模型权重、trainer state、完整
hidden-state vectors 或不可覆盖的本地 run 目录。

RecurrentGRIP v1.1.1 diagnostic cross 完成后，运行脚本会自动生成：

```text
09_results_analysis/artifacts/RecurrentGRIP_v1_1_1/<RUN_DIR_NAME>/
├── artifact_manifest.json
├── config.json
├── input_stats.json
├── environment.txt
├── cross_run_audit.json
├── context_manifests/
├── predictions_audit.jsonl
└── analysis/
```

`predictions_audit.jsonl` 保留逐题准确率、train/eval depth、adapter control、输出质量、延迟和
候选命中等字段，但移除体积较大的 `step_pooled_hidden_states`；对应的聚合 norm、cosine、
relative-delta 统计保存在 `analysis/state_dynamics*.json/csv`。

导出命令：

```bash
RUN_DIR=/absolute/path/to/completed/run \
PYTHON="$(command -v python)" \
  bash 08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29/configs/export_diagnostic_cross_artifacts.sh
```

导出器要求 `cross_run_audit.json` 为 `status=pass`，并检查 prediction count 与
`analysis/summary.json` 一致。已存在的导出目录不会被覆盖。

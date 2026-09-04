# PriorityDistill-GRIP v0.1.1 — Oracle Stage-2 Diagnostic

- **日期**：2026-09-01
- **状态**：`READY_FOR_WSL_GPU`
- **目的**：解释 v0.1 oracle smoke 的失败是“Stage 2 练习太少”，还是“Stage 1 学会了复制路径末节点的捷径”。
- **运行环境**：WSL2 + RTX 3090 + conda `guardenv`

## 和 v0.1 的区别

这个版本只运行 `oracle_priority_equal_token`，不运行 full seeds，也不实现 learned prioritizer。它保留 v0.1 的 Stage-1 oracle/equal-token 构造，但把 Stage 2 拆成独立 epoch，并在以下时间点保存 checkpoint：

1. `initial`：训练前
2. `stage1_end`：Stage 1 结束
3. `stage2_epoch1` 到 `stage2_epochN`

每个 checkpoint 都有两种评估：

- `graph_free`：真正的目标评估，只给原始问题，不给路径；
- `oracle_evidence`：诊断评估，给出该题的 gold path，用于确认模型是否会利用路径。这个条件不代表最终部署能力。

## 运行

```bash
cd 08_experiments/PriorityDistillGRIP/v0_1_1_oracle_stage2_diagnostic_2026-09-01
source /home/kieran/miniconda3/etc/profile.d/conda.sh
conda activate guardenv

CONDA_ENV=guardenv \\
RUN_ID=wsl3090_oracle_stage2_diagnostic_20260901_01 \\
bash configs/run_wsl_diagnostic.sh
```

脚本默认使用本地模型快照：

```text
/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```

如需覆盖：

```bash
MODEL_NAME_OR_PATH=/absolute/path/to/model bash configs/run_wsl_diagnostic.sh
```

## 主要产物

- `diagnostic_metrics.jsonl`：每个 checkpoint、条件和 split 的完整指标；
- `DIAGNOSTIC_REPORT.md`：简表；
- `predictions_<checkpoint>_<condition>_<split>.jsonl`：预测和 prompt；
- `checkpoints/<checkpoint>/adapter_model.pt`：本地权重，遵守 `.gitignore`，不提交；
- `run_summary.json`：环境、预算、训练和 checkpoint 审计。

## 判读规则

- `oracle_evidence` 高、`graph_free` 低：模型依赖路径，存在 shortcut；
- 随 Stage-2 epoch 增加，`graph_free` 持续上升：原 v0.1 主要是 Stage-2 不够长；
- 8 个 epoch 后仍不升：优先修改 evidence 格式，阻断答案尾节点复制，不要盲目增加训练量。

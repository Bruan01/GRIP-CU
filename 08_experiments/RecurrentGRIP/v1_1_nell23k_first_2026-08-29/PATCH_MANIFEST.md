# Patch Manifest

## Inherited from v1

- fixed-depth shared recurrent executor；
- executor-only graph-specific LoRA；
- recurrent training/evaluation runner；
- adapter correct/shuffled/none controls；
- CUDA placement and `require_cuda` guard；
- immutable per-run directories and GNU `timeout`；
- CLEGR preparation code and tests（保留但不作为当前前置步骤）。

## Added in v1.1

- `grip-exp/scripts/prepare_recurrent_nell23k.py`
- `grip-exp/tests/test_prepare_recurrent_nell23k.py`
- `configs/prepare_nell23k.sh`
- `configs/run_nell23k_smoke_wsl.sh`
- `configs/run_nell23k_pilot_wsl.sh`
- `configs/analyze_nell23k.sh`
- `configs/nell23k_pilot_qwen05b.json`
- `design/NELL23K_FIRST_PLAN.md`

## Modified in v1.1

- `grip-exp/grip/recurrent/outputs.py`
  - `true_hop` 允许为 `null`，用于 NELL23K 不可达/自查询样本。
- `grip-exp/scripts/run_recurrent_grip.py`
  - 去除 CLEGR-only 错误文案；
  - 支持可空 hop 和 NELL23K 结构元数据。
- `grip-exp/evaluation/recurrent_metrics.py`
  - 整体准确率包含 unknown-hop 样本；
  - hop/K 分析只使用 known-hop 样本；
  - 增加按 K 和 adapter control 汇总。
- `grip-exp/scripts/analyze_recurrent_results.py`
  - 新增 `k_adapter_accuracy.csv`。
- recurrent output/metrics tests
  - 覆盖 `true_hop=null`。

## Intentionally Unchanged

- `13_base_method/grip-exp/`；
- `v1_fixed_depth_2026-08-28/`；
- Original GRIP 默认 runner；
- recurrent executor 架构；
- LoRA target scope；
- adaptive halting、gate、frontier loss、routing 和跨图 meta-training 均未加入。

# RecurrentGRIP v1.1.1 — NELL23K Storage × Execution Diagnostic Cross

## Experiment ID

`RG-E3-v1.1.1-diagnostic-cross-2026-08-29`

## Goal

修复 v1.1 smoke 中 graph context 被 `max_context_samples=256` 前缀截断、实际主要包含
node declarations 的问题，然后用 `K_train ∈ {1,2}` × `K_eval ∈ {1,2}` 交叉实验判断：

1. relation edge facts 是否真正进入 graph-specific LoRA；
2. K=2 塌缩来自执行器本身，还是 train/eval recurrence depth 不匹配；
3. RecurrentGRIP fixed-depth 创新点是否值得进入更大 Pilot。

本版本是独立快照，不修改：

```text
13_base_method/grip-exp
v1_fixed_depth_2026-08-28
v1_1_nell23k_first_2026-08-29
```

## Mechanism Boundary

```text
Z(0) = E(q)
Z(k+1) = F_{phi, DeltaW_G}(Z(k))
y = D(Z(K))
```

本版本保留单个共享 decoder executor block 和 executor-only graph LoRA。只修改：

- graph-memory context selection；
- train/eval depth cross；
- prediction audit fields；
- result aggregation；
- recurrent hidden-state drift analysis。

未加入 adaptive halting、gate、step embedding、residual scaling、frontier loss、routing
或跨图 meta-training。

## Why v1.1 Smoke Is Not Yet a Mechanism Test

v1.1 的 `GenGraphContextTask` 先输出所有 node，再输出所有 edge。NELL23K 有约 20K
nodes，而 smoke 对完整列表执行 `context_samples[:256]`，因此 Stage-1 很可能没有看到
relation edge facts。v1.1 的 K=2 下降说明 recurrent execution 当前有害，但不能区分：

- graph storage 未发生；
- train/eval depth mismatch；
- repeated decoder block 的表示漂移；
- 输出格式退化。

## Stratified Context

新参数：

```text
--context_node_samples 32
--context_edge_samples 224
--context_sampling_seed 2026
```

采样策略固定为 `node_random_train_qa_anchor_edge_relation_round_robin_v2`：

1. 先锚定 64 个 train QA 对应的 exact graph facts；
2. 再用剩余 160 个 edge slots 做 relation round-robin；
3. node declarations 独立随机抽取，不再占用 edge 预算。

旧 `--max_context_samples` 保留兼容，但与新 quota 同时启用会直接报错。

每个 adapter 在训练前写出：

```text
adapters/<graph_id>/context_sampling_manifest.json
```

manifest 包含请求/实际 node、edge、relation coverage、train-QA coverage、选中事实和
SHA256，可直接审计 Stage-1 到底看到了什么。

## NELL23K Static Context Audit

对当前固定的 `64/32/64` NELL23K diagnostic 输入执行纯 Python 采样审计：

```text
unique nodes                  = 20,799
raw train edges               = 24,321
clean unique edges            = 24,310
relations                     = 198
selected nodes / edges        = 32 / 224
selected relations            = 198 / 198 (100%)
train QA exact facts covered  = 64 / 64 (100%)
train QA relation coverage    = 64 / 64 (100%)
train QA endpoints covered    = 128 / 128 (100%)
selection SHA256              = d4a9cf3f6d2deee090af5137a2b52f89f8ce335926eabbd344ece85729d72885
```

这证明 **Stage-1 的输入选择路径已经包含目标训练事实且关系覆盖完整**；它不等价于证明
LoRA 已经成功记住这些事实。参数化存储是否发生，仍必须由 correct-vs-none 的逐题配对
结果和后续 memory probe 判定。

## macOS Check

```bash
cd 08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29
bash configs/check_mac_static.sh
```

macOS 仅做 AST、Bash、JSON 与纯 Python 测试，不运行模型训练。

## WSL2 + RTX 3090 Diagnostic Cross

先拉取包含本版本的分支并准备环境/数据：

```bash
cd 08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
bash configs/prepare_nell23k.sh
```

确保输入仍是小规模 diagnostic split：

```bash
MAX_TRAIN_QUESTIONS=64 \
MAX_VALIDATION_QUESTIONS=32 \
MAX_TEST_QUESTIONS=64 \
  bash configs/prepare_nell23k.sh
```

运行不可覆盖的交叉实验：

```bash
RUN_ID=wsl3090_nell23k_diag_cross_20260829_01 \
  bash configs/run_nell23k_diagnostic_cross_wsl.sh
```

目录结构：

```text
results/runs/<RUN_ID>_nell23k_diagnostic_cross/
├── environment.txt
├── config.json
├── input_stats.json
├── train_k1/
│   ├── adapters/nell23k/context_sampling_manifest.json
│   ├── predictions.jsonl
│   └── run.log
├── train_k2/
├── predictions.jsonl
├── cross_run_audit.json
├── analysis/
└── analysis.log
```

## Analysis Outputs

```text
analysis/summary.json
analysis/train_eval_depth_accuracy.csv
analysis/split_train_k_eval_k_adapter_accuracy.csv
analysis/transition_k1_k2.csv
analysis/output_quality.json
analysis/output_quality_by_train_eval.csv
analysis/state_dynamics.json
analysis/state_dynamics_by_train_eval.csv
analysis/state_dynamics_by_split_train_eval.csv
analysis/k_adapter_accuracy.csv
analysis/split_k_adapter_accuracy.csv
analysis/hop_k_accuracy.csv
```

每条 prediction 额外包含：

```text
recurrent_train_k
generated_token_count
ended_with_eos
response_in_candidates
step_pooled_hidden_states
```

## Git-Trackable Audit Export

根 `.gitignore` 会忽略完整 `results/runs/`，因此 diagnostic runner 在分析完成后自动导出
小型审计产物到：

```text
09_results_analysis/artifacts/RecurrentGRIP_v1_1_1/<RUN_DIR_NAME>/
```

导出内容包含 config、environment、cross audit、两个 context manifests、全部 analysis
JSON/CSV，以及移除完整 `step_pooled_hidden_states` 后的逐题 `predictions_audit.jsonl`。
state dynamics 的聚合 norm/cosine/relative-delta 仍完整保留。导出器会校验 cross audit、
prediction count、summary count，并生成包含 SHA256 的 `artifact_manifest.json`。

手动重新导出尚未导出的 run：

```bash
RUN_DIR=/absolute/path/to/completed/run \
PYTHON="$(command -v python)" \
  bash configs/export_diagnostic_cross_artifacts.sh
```

已存在的导出目录不会被覆盖。

## Innovation Assessment

当前创新性与可证伪边界见 `design/INNOVATION_ASSESSMENT.md`。核心结论是：固定深度
direct recurrence 本身不是足够的新方法；本版本用于判断 graph-specific adapter 是否能成为
稳定、可重复应用的状态转移算子，并为下一版 recall/normalization/gating 机制提供触发条件。

## Go / No-Go

详细规则见 `design/DIAGNOSTIC_CROSS_PLAN.md`。当前状态：

```text
工程执行链：已通过 v1.1 smoke
fixed-depth recurrence 有效：未证明
graph storage 输入：v1.1.1 静态审计通过；参数化存储效果仍待 WSL cross
是否直接运行两小时 Pilot：否
下一步：只运行 v1.1.1 diagnostic cross
```

只有在 edge manifest 有效、correct adapter 出现可复现信号、且 K=2 行为能被
`K_train` 或后续稳定化机制解释时，才扩大数据和随机种子。

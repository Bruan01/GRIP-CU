# Patch Manifest

## Inherited

- v1 fixed-depth shared recurrent executor；
- executor-only graph-specific LoRA；
- correct/shuffled/none adapter controls；
- NELL23K train/valid/test adapter；
- WSL2 RTX 3090 setup/test/run path；
- Qwen decoder forwarding、answer parser、CUDA placement 与 timeout 修复。

## Added in v1.1.1

- `grip-exp/recurrent_context_sampling.py`
  - 纯 Python node/edge 分层采样；
  - train-QA exact facts 优先锚定；
  - 剩余 edge budget 按 relation round-robin；
  - duplicate/self-loop 清理；
  - QA fact/relation/entity coverage 与 SHA256 manifest。
- `grip-exp/tests/test_context_sampler.py`
- `grip-exp/recurrent_cross_audit.py`
  - 检查两个 train-depth run 使用相同 context hash、完整矩阵和相同 question set。
- `grip-exp/tests/test_recurrent_cross_audit.py`
- `configs/run_nell23k_diagnostic_cross_wsl.sh`
- `configs/nell23k_diagnostic_cross_qwen05b.json`
- `configs/export_diagnostic_cross_artifacts.py`
  - 校验 cross-run audit 与 prediction/summary count；
  - 导出 config、environment、context manifests 和 analysis；
  - 移除完整 hidden-state vectors，保留逐题审计字段；
  - 生成含 SHA256 的 artifact manifest。
- `configs/export_diagnostic_cross_artifacts.sh`
- `grip-exp/tests/test_export_diagnostic_artifacts.py`
- `design/DIAGNOSTIC_CROSS_PLAN.md`

## Modified in v1.1.1

- `grip-exp/arguments/recurrent_args.py`
  - 新增 context node/edge quota 与 seed；
  - 禁止 legacy prefix cap 与 stratified sampler 同时启用。
- `grip-exp/scripts/run_recurrent_grip.py`
  - stratified context formatting；
  - 训练前保存 `context_sampling_manifest.json`；
  - prediction 写入 train depth、token/EOS/candidate audit fields。
- `grip-exp/grip/recurrent/outputs.py`
  - 新增 `recurrent_train_k`、`generated_token_count`、`ended_with_eos`、
    `response_in_candidates`。
- `grip-exp/evaluation/recurrent_metrics.py`
  - train/eval depth cross buckets；
  - K1→K2 paired transitions；
  - empty/candidate/EOS/token/response-length 统计；
  - recurrent hidden-state norm/cosine/relative-delta 统计。
- `grip-exp/scripts/analyze_recurrent_results.py`
  - 新增 cross、transition、output-quality 和 state-dynamics artifacts。
- `configs/check_mac_static.sh`
  - 执行新增纯 Python测试。
- `configs/analyze_nell23k.sh`
  - 优先发现 diagnostic cross run。

## Intentionally Unchanged

- `13_base_method/grip-exp/`；
- `v1_fixed_depth_2026-08-28/`；
- `v1_1_nell23k_first_2026-08-29/`；
- recurrent executor architecture；
- LoRA target scope；
- adaptive halting、gate、step embedding、residual scaling、frontier loss、routing。

## Verified NELL23K Context Invariant

固定 `64/32/64` diagnostic 输入、node/edge quota=`32/224`、seed=`2026` 时：

- 64/64 train QA exact facts 被选入；
- 198/198 relations 被覆盖；
- 128/128 train QA endpoints 出现在选中 context 中；
- selection SHA256：`d4a9cf3f6d2deee090af5137a2b52f89f8ce335926eabbd344ece85729d72885`。

该 invariant 用于确认 storage 输入有效，不把“被输入”误写成“已被参数记忆”。

# EntityConstrained-GRIP v0.1 — Validation-First Global Entity Decoder

- **日期**：2026-09-04
- **状态**：`READY_TO_PUSH_VALIDATION_ONLY_E03`
- **定位**：E03，L6 查询、解码与评测机制；冻结既有 LoRA，只替换 decoder/readout。
- **执行限制**：完整静态门禁已通过；当前仍未 commit、未 push、未启动 WSL GPU，等待明确指令后再提交与推送。

## 研究策略

采用 **官方 GRIP 配置作为最终论文锚点 + 0.5B 冻结 checkpoint 先做机制证伪**：

1. **Phase A**：只在 validation 上比较历史 D0 自由生成 artifact 与 D1 train-KG 全局实体 trie，不重新训练，不读取 test。
2. **D2 follow-up**：仅当两个 direct checkpoint 的 D1 平均 canonical EM 提升达到 `+2 pp` 且通过稳健性门槛后运行。
3. **Phase B**：只有机制验证通过后，才回到官方 Qwen2.5-7B、LoRA rank/alpha `4/8`、三随机种子设置。

0.5B Phase-A 结果只能作为机制证据，不能替代官方 GRIP 主表。

## 首轮 checkpoint

| checkpoint | 角色 | 选择规则 | D0 来源 |
|---|---|---|---|
| `direct_answer_only_seed43` | primary direct | validation-selected epoch 5 | 冻结 graph-free validation predictions |
| `direct_answer_only_seed44` | primary direct | validation-selected epoch 4 | 冻结 graph-free validation predictions |
| `more_qa_equal_token_seed42` | reference | 既有等 token More-QA | 冻结 validation predictions |

`joint_trace_answer_seed43/44` 仅登记为 secondary，不进入首轮 gate。

所有 checkpoint 与 D0 artifact 均在 config 中登记 SHA256。checkpoint 权重由原 WSL 机器保存，Git 不追踪；D0 artifact 在本地静态审计时已做 hash、task_id、answer 和行数校验。

## Decoder registry

| Decoder | 定义 | 使用时机 | 论文地位 |
|---|---|---|---|
| D0 | 复用对应 checkpoint 的历史自由生成 validation artifact | 首轮 | 基线；不重复生成 |
| D1 | 使用 train graph 20,789 entities 的全局 token trie 约束生成 | 首轮 | 主证伪机制 |
| D2 | 全局实体 sequence log-likelihood 排序，sum/mean 仅在 validation 选择 | D1 过 gate 后 | follow-up |
| D3 | gold + 3 个 train-only same-depth distractors | 单独 diagnostic mode | oracle diagnostic，不进 gate |

## 严格边界

- 实体词表只从 `13_base_method/grip-exp/data/raw_datasets/nell23k/train.txt` 构建；
- 所有 query 共用同一份 20,789 实体全局词表；
- Phase-A runner 只打开 train 与 validation，明确记录 `test_file_opened=false`；
- D0/D1/D2 不读取 query-specific subgraph、neighbors、gold path、gold trace、gold type 或 gold candidate；
- D3 使用 gold-containing candidate pool，因此始终是 diagnostic-only；
- Original GRIP `13_base_method/grip-exp/` 保持只读。

静态 setup audit 仍可读取 test 以验证历史 split 隔离和覆盖率；这不属于 Phase-A GPU 推理流程。

## Error-transition probe

D0/D1 prediction 按 `task_id` 严格一一对齐，并输出：

- `d0_invalid_to_d1_correct`；
- `d0_valid_wrong_to_d1_correct`；
- `d0_correct_to_d1_wrong`；
- `d0_wrong_unchanged`；
- 以及 invalid/valid-wrong/correct 的完整去向计数与比例。

这用于区分“格式合法化”与“真实语义纠错”。

## 预注册 gate

只有以下条件全部满足，才输出 `PRELIMINARY_GO_D2`：

1. 至少两个 primary direct checkpoints 完成 D0/D1 validation 对比；
2. 两个 direct checkpoints 的 D1 over D0 **平均 canonical EM ≥ +2 pp**；
3. 每个 direct checkpoint 的 raw 与 canonical 增益均不为负；
4. novel-composition canonical EM 的平均下降不超过 `1 pp`。

valid-entity rate 提升单独报告，但不能替代 EM。否则输出 `STOP_DECODER_PRIMARY`，不跑 D2，不进入 Phase B。

## macOS 静态检查

```bash
cd /Users/mac/CODE/GRIP/ai_research_workflow/08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04
bash configs/check_static.sh
```

2026-09-04 本地复审结果：

- `Ran 40 tests` / `OK`；
- `ENTITY_VOCABULARY_READY entities=20789 test_coverage=1.000000`；
- `STATIC_READY_D0_ARTIFACTS_VERIFIED_REMOTE_CHECKPOINTS_OPTIONAL`；
- `STATIC_CHECKS_PASSED`；
- `bash -n configs/*.sh` 与 `python3 -m py_compile entity_decoder/*.py scripts/*.py` 通过；
- 旧协议残留搜索为空；Phase-A runner 只加载 train/validation；Original GRIP diff 为空。

当前结论仅为“代码与协议可推送”，不代表已经产生 GPU 实验结果。

## WSL 命令（仅在本地审计决定可推送后使用）

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04
export PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
export MODEL_NAME_OR_PATH=/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
RUN_ID=wsl3090_entity_decoder_validation_01 RUN_MODE=phase_a bash configs/run_wsl_smoke.sh
```

只有 `REPORT.md` 为 `PRELIMINARY_GO_D2` 时：

```bash
RUN_ID=wsl3090_entity_decoder_validation_01 RUN_MODE=d2 bash configs/run_wsl_smoke.sh
```

D3 必须显式单独运行：

```bash
RUN_ID=wsl3090_entity_decoder_validation_01 RUN_MODE=diagnostic bash configs/run_wsl_smoke.sh
```

## 结果

```text
results/runs/<RUN_ID>/REPORT.md
results/runs/<RUN_ID>/suite_summary.json
results/runs/<RUN_ID>/phase_a/<checkpoint>/run_summary.json
results/runs/<RUN_ID>/phase_a/<checkpoint>/runtime_audit.json
results/runs/<RUN_ID>/phase_a/<checkpoint>/predictions_D{0,1}_validation.jsonl
results/runs/<RUN_ID>/phase_a/<checkpoint>/d0_to_d1_validation_transitions.json
```

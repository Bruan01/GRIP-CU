# TODO — EntityConstrained-GRIP

更新日期：2026-09-04

当前执行版本：

```text
08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/
```

## 当前决策

```text
READY_TO_PUSH_VALIDATION_ONLY_E03
```

采用：

```text
官方 GRIP Qwen2.5-7B 配置作为最终论文锚点
+
既有 Qwen2.5-0.5B checkpoints 做 validation-only decoder 机制证伪
```

本地静态复审与变更边界审计已通过；当前仍未 commit、未 push、未运行 WSL，等待明确提交/推送指令。

## 已完成

- [x] Original GRIP `13_base_method/grip-exp/` 保持只读。
- [x] 构建 NELL23K train graph 全局实体词表：20,789 entities。
- [x] 将 primary checkpoints 改为 validation-selected direct seed43/44。
- [x] 将 More-QA seed42 登记为 reference，将 joint seed43/44 降为 secondary。
- [x] 首轮协议改为 validation-only D0 artifact reuse + D1 global trie。
- [x] 为三个首轮 D0 artifacts 登记并校验 SHA256、152 task_id 与 answer 对齐。
- [x] 增加 D0→D1 error-transition decomposition。
- [x] gate 改为 direct 平均 canonical EM `≥ +2 pp`、每 checkpoint raw/canonical 非负、novel-composition 平均下降不超过 `1 pp`。
- [x] valid-rate 明确降为解释指标，不能替代 semantic EM。
- [x] D2 改为只有 `PRELIMINARY_GO_D2` artifact 才可运行。
- [x] D3 改为显式 diagnostic mode，不进入 gate。
- [x] Phase-A runner 只打开 train/validation，不读取 test。
- [x] 单元测试由 27 个扩展到 40 个。

## 当前待办

- [x] 运行 `bash configs/check_static.sh` 完整静态门禁：40 tests OK，`STATIC_CHECKS_PASSED`。
- [x] 审计 Git diff：Original GRIP diff 为空；无关 untracked 文件未修改或删除。
- [x] 将状态更新为 `READY_TO_PUSH_VALIDATION_ONLY_E03`。
- [ ] 仅在用户明确要求后 commit/push。
- [ ] 推送后在原 WSL2 + RTX 3090 执行 `RUN_MODE=phase_a`。
- [ ] 若 gate 为 `STOP_DECODER_PRIMARY`：停止 D2/D3/Phase B，回传 artifacts 做结果审计。
- [ ] 若 gate 为 `PRELIMINARY_GO_D2`：只补跑 validation D2。
- [ ] 仅在后续机制证据成立后，创建新的官方 Qwen2.5-7B 三 seed Phase-B 版本。

## WSL 入口（当前仅待用）

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04
export PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
export MODEL_NAME_OR_PATH=/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
RUN_ID=wsl3090_entity_decoder_validation_01 RUN_MODE=phase_a bash configs/run_wsl_smoke.sh
```

条件式 D2：

```bash
RUN_ID=wsl3090_entity_decoder_validation_01 RUN_MODE=d2 bash configs/run_wsl_smoke.sh
```

## 结果入口

```text
results/runs/wsl3090_entity_decoder_validation_01/REPORT.md
results/runs/wsl3090_entity_decoder_validation_01/suite_summary.json
results/runs/wsl3090_entity_decoder_validation_01/phase_a/<checkpoint>/runtime_audit.json
results/runs/wsl3090_entity_decoder_validation_01/phase_a/<checkpoint>/run_summary.json
results/runs/wsl3090_entity_decoder_validation_01/phase_a/<checkpoint>/d0_to_d1_validation_transitions.json
```

# 下一步给 WSL Codex 的完整 Prompt

> 使用条件：macOS 静态审计已通过，且用户已明确执行 commit/push；当前文件只是待用运行说明。

```text
你现在在 WSL2 + RTX 3090 上执行 EntityConstrained-GRIP v0.1 validation-only 机制实验。

仓库：/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU
分支：wsl/nell23k-smoke-20260829
实验目录：08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04
Python：/home/kieran/miniconda3/envs/guardenv/bin/python
模型：/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775

约束：
1. 不修改 13_base_method/grip-exp，不训练 checkpoint。
2. 首轮只运行 validation；不要运行或读取 test predictions。
3. D0 必须复用 config 登记的历史 artifact，不重新自由生成。
4. D1 只使用 train-KG 的 20,789 entity global trie，不使用 query-specific graph/candidate/gold trace。
5. primary 为 direct seed43/44，More-QA seed42 为 reference，joint 不进入首轮。
6. D3 是 gold-containing diagnostic，不进入 gate。
7. 不覆盖已有 RUN_ID。

执行：
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04
export PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
export MODEL_NAME_OR_PATH=/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
RUN_ID=wsl3090_entity_decoder_validation_01 RUN_MODE=phase_a bash configs/run_wsl_smoke.sh

完成后检查：
- results/runs/wsl3090_entity_decoder_validation_01/REPORT.md
- results/runs/wsl3090_entity_decoder_validation_01/suite_summary.json
- phase_a/<checkpoint>/runtime_audit.json
- phase_a/<checkpoint>/run_summary.json
- phase_a/<checkpoint>/d0_to_d1_validation_transitions.json

只有 REPORT.md 的 decision 为 PRELIMINARY_GO_D2 时运行：
RUN_ID=wsl3090_entity_decoder_validation_01 RUN_MODE=d2 bash configs/run_wsl_smoke.sh

若 decision 为 STOP_DECODER_PRIMARY，停止 D2、D3 和 Phase B，并回传完整结果目录用于本地审计。

回传：GPU、torch/transformers 版本、checkpoint 与 D0 artifact hash 状态、D0/D1 validation raw/canonical/valid/novel-composition 指标、error transitions、延迟、峰值显存和 gate decision。
```

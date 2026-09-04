# PriorityDistill-GRIP v0.1.2 — Controlled Protocol

- **日期**：2026-09-01
- **状态**：`READY_FOR_WSL_GPU`
- **目的**：公平比较 `direct_answer_only` 与 `oracle_two_stage`，验证 Stage 1 是否给 graph-free 回答带来额外收益。
- **环境**：WSL2 + RTX 3090 + conda `guardenv`

## 两个协议

### direct_answer_only

从初始 Qwen 开始，始终训练：

```text
问题 → 答案
```

### oracle_two_stage

从同一个初始 Qwen 开始：

```text
Stage 1: 问题 + gold path → 答案
Stage 2: 问题 → 答案
```

## 公平控制

两种协议使用同一份数据、同一 LoRA、同一 seed、同一 token 累计预算和同一优化器预算分段。两种协议的 Stage-1 等价预算都按 `all_paths_equal_token` 的 token 数确定；之后都执行 8 个相同 token 数的 answer-only segment。direct 协议把第一段也替换成 answer-only，因此它不会执行真正的 Stage 1。

模型选择只看 `graph_free_validation`；`selected_test_metrics.json` 中的 test 是选点后读取的 held-out 结果，不能用 test 反选 checkpoint。

## 运行前检查

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/PriorityDistillGRIP/v0_1_2_controlled_protocol_2026-09-01
source /home/kieran/miniconda3/etc/profile.d/conda.sh
conda activate guardenv
bash configs/check_static.sh
```

## 运行命令

先运行 direct baseline，再运行 oracle two-stage。为两个协议使用不同 RUN_ID。

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/PriorityDistillGRIP/v0_1_2_controlled_protocol_2026-09-01
source /home/kieran/miniconda3/etc/profile.d/conda.sh
conda activate guardenv

CONDA_ENV=guardenv \
CONFIG_PATH=configs/direct_answer_only.json \
RUN_ID=wsl3090_control_direct_20260901_01 \
bash configs/run_controlled_wsl.sh

CONDA_ENV=guardenv \
CONFIG_PATH=configs/oracle_two_stage.json \
RUN_ID=wsl3090_control_oracle2_20260901_01 \
bash configs/run_controlled_wsl.sh
```

默认使用本地缓存：

```text
/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```

可以通过 `MODEL_NAME_OR_PATH` 覆盖。权重位于 `results/runs/`，由 `.gitignore` 忽略；metrics、predictions、logs、report 可以提交。

## 进度条

运行时会显示三层进度信息：

- `direct_answer_only overall` / `oracle_two_stage overall`：整个协议的 segment 进度；
- `<segment> train`：当前 segment 的 batch 进度，并显示最近 loss 和 optimizer step；
- `<condition> eval <checkpoint>`：当前 checkpoint 的评估 batch 进度。

如果需要关闭进度条，可临时修改配置中的：

```json
"runtime": {"disable_progress": true}
```

# PriorityDistill-GRIP v0.1.4 — Explicit Path Selection and Anti-Copy

- **日期**：2026-09-01
- **状态**：`READY_FOR_WSL_GPU`
- **目的**：检验 Stage 1 的路径信息是否能帮助模型学习候选路径选择，而不是只复制路径末节点。
- **环境**：WSL2 + RTX 3090 + conda `guardenv`

## 为什么做这个版本

前一版本中，Oracle Stage 1 的 gold path 最后一个节点就是答案。模型可以直接把路径末节点复制到输出，并不代表它已经把图上的关系内化。v0.1.3 把“选哪条路径”和“输出答案”拆成两个可测量目标，并保留 anti-copy 条件，并把 Stage 2 学习率降为 `3e-4`。

## 协议

| 配置 | Stage 1 | Stage 2 | 目的 |
|---|---|---|---|
| `direct_answer_only` | 无路径 | 纯 answer-only | 基线 |
| `oracle_two_stage` | 单条 gold path | 纯 answer-only | 当前 Oracle 对照 |
| `candidate_index_two_stage` | 1 条 gold + 3 条 distractor，随机顺序 | 纯 answer-only | 测试候选路径匹配 |
| `candidate_index_anti_copy` | 4 条候选路径，所有终点隐藏，另给无关联 endpoint 列表 | 纯 answer-only | 测试去除终点直接复制后的匹配 |
| `candidate_index_anti_copy_replay` | 同上 | 80% answer-only + 20% masked evidence replay | 减轻 Stage 1 到 Stage 2 的遗忘 |

所有 distractor 都满足：

- 与题目相同 hop 数；
- 答案不同；
- 只从 train split 选取；
- 不标记哪条是 gold；
- 每题候选顺序确定性随机化。

Anti-copy 条件把每条路径的 terminal entity 替换为 `<MASKED_TERMINAL>`，然后把四个可能 endpoint 作为无关联列表提供。这样答案仍然是一个定义清楚的选择题，但不能从某条路径的最后一个 token 直接复制答案。它是机制诊断，不保证一定比 full-path 条件更容易。

## 公平控制

- 相同 Qwen 本地快照、数据划分、LoRA 配置和 seed；
- Stage 1 的 input-token 预算按 full all-path reference 对齐；
- 每个 Stage 2 epoch 的 input-token 预算按 answer-only 对齐；
- anti-copy replay 的总 Stage 2 input-token 预算仍与 answer-only 对齐；
- 同时记录 input tokens、supervised tokens、实际 optimizer steps；
- checkpoint 只按 `graph_free_validation` 选择，之后读取 test；
- 测试时不访问路径或图。

## 静态检查

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/PriorityDistillGRIP/v0_1_4_explicit_path_selection_2026-09-01
source /home/kieran/miniconda3/etc/profile.d/conda.sh
conda activate guardenv
bash configs/check_static.sh
```

## 模型缓存

默认使用：

```text
/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```

可用 `MODEL_NAME_OR_PATH` 覆盖。

## 进度条

运行时有三层进度条：整体 segment、当前训练 batch、当前 checkpoint evaluation batch。配置中的 `runtime.disable_progress` 默认是 `false`。

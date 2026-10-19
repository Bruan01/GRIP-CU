# ErrorMemory-9 + InfoNCE for original GRIP

本目录是当前主线的唯一方法实验目录。它只在原始 GRIP 的候选关系学习阶段加入：

1. 冻结 B1 在训练问题上的生成错误，构建 gold-conditioned ErrorMemory；
2. 用 B1 的候选分数违例度和 ErrorMemory 组合采样 9 个边界负样本；
3. 在原始 GRIP 的生成损失之外加入候选关系 InfoNCE。

当前分支同时提供一个结构反事实候选分支 `structure_counterfactual`：它在训练图
关系画像相近、但对当前实体对不属于 train-known fact 的关系中，结合冻结 B1 的
分数违例度采样 9 个候选。该分支复用原始 GRIP 的 prompt、生成损失和 InfoNCE，
只替换冻结的候选关系 bank。

这里没有递归执行器、递归深度、`recurrent_questions` 数据契约或 RecurrentGRIP 运行时依赖。原始 GRIP 代码通过 `13_base_method/grip-exp` 子模块提供。

## 方法契约

训练关系候选池为：

```text
P_q = R_train \ {gold(q) ∪ T_train(head(q), tail(q))}
u(q,r) = alpha * E(gold(q), r) + beta * V_B1(q,r)
p(r|q) = rho / |P_q| + (1-rho) * softmax(u(q,r) / tau)
```

默认参数为 `K=9`、`rho=1/3`、`alpha=1`、`beta=1`、`tau=0.25`。所有错误记忆和分数只来自训练 split；冻结后训练只读取 JSONL bank，不在线重采样。Random-9 使用同一训练关系词表、同一 `K`、同一训练过滤和同一训练预算。

结构反事实分支的候选 utility 为：

```text
s(q,r) = relation_profile_similarity(train_graph, gold(q), r)
u(q,r) = alpha * s(q,r) + beta * V_B1(q,r)
p(r|q) = rho / |P_q| + (1-rho) * softmax(u(q,r) / tau)
```

这里的“反事实”是严格的 train-only 操作契约：在已观测实体对上替换关系标签，
并过滤 train split 中已经存在的关系；它不把开放世界知识图谱中的未知事实宣称
为绝对真或假。bank 冻结后训练不再在线重采样。

## 目录

- `src/hard_negative_grip/error_memory_sampling.py`：冻结 ErrorMemory-9 bank。
- `src/hard_negative_grip/counterfactuals.py`：冻结结构反事实关系 bank。
- `src/hard_negative_grip/listed_training.py`：原始 GRIP 生成损失 + 候选 InfoNCE。
- `src/hard_negative_grip/task_file.py`：标准 GRIP task/graph record 与 manifest 校验。
- `scripts/mine_rollout_hard_negatives.py`：冻结 B1 训练问题生成错误。
- `scripts/score_relation_candidates_offline.py`：冻结 B1 对训练关系词表的分数。
- `scripts/mine_error_memory_negatives.py`：CPU 冻结 ErrorMemory-9。
- `scripts/build_structure_counterfactual_bank.py`：CPU 冻结结构反事实 bank。
- `scripts/verify_error_memory_wiring.py`：CPU 接线和泄漏检查。
- `scripts/train_listed_contrastive.py`：只负责原始 GRIP 的候选关系学习阶段及评估。
- `configs/run_error_memory_wsl.sh`：WSL 分阶段入口，默认只做 preflight。
- `configs/error_memory9_pair.json`、`configs/random9_pair.json`：主方法与 Random-9 控制协议。

## 数据与基线

- baseline code：`13_base_method/grip-exp`；
- raw NELL23K：`04_benchmark/data/nell23k/raw`；
- raw FB15K237-2：`04_benchmark/data/fb15k237_2/raw`；关系文本映射为 `rel2text.json`；
- 评估文件：对应数据集的标准 `processed_test.json`，记录包含 `title/graph/questions/answers/id`；
- 训练 task JSON：包含原始 GRIP 的 `context_samples` 和 `qa_samples`。正式训练必须显式提供完整文件，不能用 smoke 子集替代。

## WSL 命令

以下命令只检查文件和环境，不启动 GPU：

```bash
cd /path/to/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
GRIP_BASELINE_DIR=/path/to/GRIP-CU/13_base_method/grip-exp \
INPUT_FILE=/path/to/GRIP-CU/04_benchmark/data/nell23k/tasks/grip_nell23k_tasks.json \
EVAL_FILE=/path/to/GRIP-CU/04_benchmark/data/nell23k/processed/processed_test.json \
RAW_DIR=/path/to/GRIP-CU/04_benchmark/data/nell23k/raw \
MODEL_CACHE_DIR=/mnt/e/GRIP-CU/model_cache \
S1_ADAPTER=/path/to/GRIP-CU/13_base_method/grip-exp/artifacts/nell23k_paper_7b/lora_adapter \
B1_ADAPTER=/path/to/GRIP-CU/13_base_method/grip-exp/artifacts/nell23k_paper_7b/lora_adapter \
STEP=preflight bash configs/run_error_memory_wsl.sh
```

FB15K237-2 使用同一入口，将 `INPUT_FILE`、`EVAL_FILE`、`RAW_DIR` 改成 `04_benchmark/data/fb15k237_2/`，并额外设置 `RELATION_TEXT_FILE=$RAW_DIR/rel2text.json`。Qwen2.5-7B 基座位于 `/mnt/e/GRIP-CU/model_cache/`；历史 FB 目录位于 `/mnt/e/GRIP-CU/artifacts/info_nce_fb15k237_qwen7b_20260921/`，但当前只保留 adapter 配置/Tokenizer，未发现完整 `adapter_model.safetensors`。FB GPU 步骤必须显式传入已恢复或新训练的完整 `S1_ADAPTER`/`B1_ADAPTER`，运行前在 `environment.txt` 中记录实际路径和 hash。

经审核后按以下顺序执行：

```bash
STEP=rollout bash configs/run_error_memory_wsl.sh  # 冻结 B1 训练错误，GPU
STEP=score bash configs/run_error_memory_wsl.sh    # 冻结 B1 候选分数，GPU
STEP=freeze bash configs/run_error_memory_wsl.sh   # 冻结 ErrorMemory-9，CPU
STEP=wire bash configs/run_error_memory_wsl.sh     # 检查完整 task 的接线，CPU

ALLOW_TRAIN=1 STEP=train \
  SWANLAB_PROJECT=grip-negative-sampling \
  SWANLAB_GROUP=error-memory-k9 \
  bash configs/run_error_memory_wsl.sh
```

`STEP=train` 只在显式设置 `ALLOW_TRAIN=1` 后运行，并通过 `--listed_negative_source error_memory` 和 `--negative_sampling_mode error_memory` 启用主方法。SwanLab 是可选的；设置 `SWANLAB_PROJECT` 后记录训练配置、阶段 loss、评估摘要和运行状态。

结构反事实分支跳过 rollout，但仍需要先用冻结 B1 完成 train-only candidate score：

```bash
STEP=score NEGATIVE_SOURCE=structure_counterfactual bash configs/run_error_memory_wsl.sh
STEP=freeze NEGATIVE_SOURCE=structure_counterfactual bash configs/run_error_memory_wsl.sh
STEP=wire NEGATIVE_SOURCE=structure_counterfactual bash configs/run_error_memory_wsl.sh
ALLOW_TRAIN=1 STEP=train NEGATIVE_SOURCE=structure_counterfactual \
  SWANLAB_PROJECT=grip-negative-sampling \
  SWANLAB_GROUP=structure-counterfactual-k9 \
  bash configs/run_error_memory_wsl.sh
```

训练入口会把完整 `run_config.json` 传给 SwanLab，并额外记录扁平化的数值超参数快照，
包括学习率、LoRA、batch/accumulation、epoch、seed、InfoNCE 权重/温度、负样本
来源以及结构采样的 `uniform_mix/structure_weight/violation_weight/temperature`。

## 论文叙事边界

主比较是：原始 GRIP/B1、Random-9 + InfoNCE、ErrorMemory-9 + InfoNCE、
Structure-Counterfactual-9 + InfoNCE。当前实现首先覆盖关系候选型 task；在四数据集
正式实验前，仍需为 WN18RR/CLEGR 完成对应 task 生成与 runner adapter，并通过同一
train-only bank/wiring gate。实验完成前不能声称提升、解决 OOV 或提出通用 KG 负采样理论。

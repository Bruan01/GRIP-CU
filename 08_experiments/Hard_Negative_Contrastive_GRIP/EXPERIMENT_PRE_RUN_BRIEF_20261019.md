# 实验前审核报告：ErrorMemory-9 + InfoNCE

日期：2026-10-19
当前分支：`codex/boundary-negative-recurrentgrip`
分支关系：从 `origin/main` 分叉，与 `origin/recurrentgrip-nell23k-20260904` 同级
当前状态：只完成算法落地和静态检查入口；没有启动 GPU、训练或 SwanLab 在线实验。

## 1. 当前的基线问题是什么？

原始 GRIP 在候选关系学习阶段用固定的候选集合做生成式训练/候选打分，但候选负样本没有利用模型自身的错误结构：

- 训练问题上 B1 已经暴露出的 gold-conditioned 混淆没有进入后续学习；
- 只按当前分数选 hard negative 容易把一次性高分误差当成稳定信号；
- 完全均匀采样又可能把训练预算放在远离决策边界的关系上。

这里的“基线”是原始 GRIP 的生成式 graph-context/QA 训练和 generation-only B1，不是递归执行器。

## 2. 我们的解决方法是什么？

只改候选关系学习阶段，在原始生成损失外加入候选级 InfoNCE：

1. 冻结 B1，在训练 split 的关系问题上做 greedy free generation，记录错误；
2. 按 gold relation 构建 ErrorMemory，统计 B1 把该 gold 混淆成哪些候选；
3. 对同一问题的训练关系词表候选计算 B1 score violation；
4. 排除同一实体对在训练图中已知的真实关系；
5. 用 `K=9`、均匀概率下限和 ErrorMemory/violation softmax 无放回采样；
6. 冻结 JSONL bank，在原始 GRIP Stage 2 候选关系学习中读取它，计算 InfoNCE。

默认采样策略为：

```text
u(q,r) = alpha E(gold(q), r) + beta V_B1(q,r)
p(r|q) = rho / |P_q| + (1-rho) softmax(u(q,r) / tau)
```

默认 `K=9`、`rho=1/3`、`alpha=1`、`beta=1`、`tau=0.25`、`lambda_candidate=1.0`。

## 3. 我们的假设是什么？

主要假设：冻结 B1 的训练集错误记忆与 score violation 的交集，比同预算 Random-9 更接近候选决策边界，因此在相同的生成目标、Stage 2 轮数、批量、梯度累积和随机种子下，能够改善候选关系辨别，并且不牺牲自由生成。

可证伪预测：

- ErrorMemory-9 的 closed-set relation MRR/Hits 应优于 Random-9；
- 若只有候选排序改善而 free-generation EM 不变，结论只能写成候选辨别改善；
- 若 ErrorMemory-9 不优于 Random-9，说明当前错误记忆/违例定义没有提供额外训练信号；
- 所有结论都必须同时报告 train-only provenance、候选有效率、OOV/错误切片和完整训练步数。

## 4. 参考哪些文献得到的方法？

- GRIP：作为原始 graph-context 生成式基线和候选关系学习位置；
- InfoNCE / noise-contrastive estimation：作为候选决策集上的 softmax 对比约束；
- KG 负采样与 hard-negative mining：提供“从结构/模型信号构造困难但合法负样本”的一般设计依据；
- 本项目此前的 KG 负采样梳理：将模型错误记忆、分数违例和均匀概率下限组合成可审计的 ErrorMemory-9，而不是宣称提出通用采样理论。

本报告不把 RecurrentGRIP 的递归机制作为方法来源或实验机制。

## 5. 涉及基线的哪个机制？

涉及原始 GRIP 的候选关系学习阶段：候选关系 continuation score、generation loss 和 LoRA adapter 训练流程保持不变；只额外向 Stage 2 加入候选关系 InfoNCE。Stage 1 graph-context 学习和 generation-only B1 是控制路径。

## 6. 涉及数据集是什么？

- 数据集：NELL23K；
- 原始数据：`13_base_method/grip-exp/data/raw_datasets/nell23k/{train,valid,test}.txt`；
- 训练输入：完整原始 GRIP task JSON，必须包含 `context_samples` 和 `qa_samples`；
- 评估输入：原始 GRIP 生成的标准 `processed_test.json`，记录必须包含 `title`、`graph`、`questions`、`answers`、`id`；
- 负样本过滤：只使用 `train.txt`，避免 valid/test KG 事实泄漏；
- 正式训练不得用 smoke 子集替代完整 task file。

## 7. 涉及的方法文件是什么？

- `src/hard_negative_grip/error_memory_sampling.py`：ErrorMemory-9 冻结算法；
- `src/hard_negative_grip/listed_training.py`：候选 InfoNCE trainer；
- `src/hard_negative_grip/task_file.py`：标准 GRIP task/graph record、回答解析、manifest 校验；
- `src/hard_negative_grip/decode_io.py`：标准 graph record 展开与断点续评估；
- `scripts/mine_rollout_hard_negatives.py`：B1 训练集错误；
- `scripts/score_relation_candidates_offline.py`：B1 训练集关系词表分数；
- `scripts/mine_error_memory_negatives.py`：CPU 冻结 bank；
- `scripts/verify_error_memory_wiring.py`：接线、ID、K 值和泄漏检查；
- `scripts/train_listed_contrastive.py`：原始 GRIP 训练/评估入口，baseline 路径为 `13_base_method/grip-exp`；
- `configs/run_error_memory_wsl.sh`：WSL 分阶段运行和 SwanLab 参数入口；
- `configs/error_memory9_pair.json`、`configs/random9_pair.json`：主方法和控制组协议。

## 8. 运行命令是什么？

先把本地路径替换成实际 WSL 路径；下面第一步只读：

```bash
cd /path/to/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP

GRIP_BASELINE_DIR=/path/to/GRIP-CU/13_base_method/grip-exp \
INPUT_FILE=/path/to/grip_nell23k_tasks.json \
EVAL_FILE=/path/to/processed_test.json \
STEP=preflight bash configs/run_error_memory_wsl.sh
```

审核通过后：

```bash
STEP=rollout bash configs/run_error_memory_wsl.sh
STEP=score bash configs/run_error_memory_wsl.sh
STEP=freeze bash configs/run_error_memory_wsl.sh
STEP=wire bash configs/run_error_memory_wsl.sh

ALLOW_TRAIN=1 STEP=train \
  SWANLAB_PROJECT=grip-negative-sampling \
  SWANLAB_GROUP=error-memory-k9 \
  bash configs/run_error_memory_wsl.sh
```

训练入口会使用 `--listed_negative_source error_memory`、`--negative_sampling_mode error_memory` 和 `--error_memory_manifest`。Random-9 控制需要使用同一原始 GRIP 训练入口、同一训练预算和 `--listed_negative_source train_graph`。

## 9. 预期是什么？

实验前只允许写成方向性预期：ErrorMemory-9 应在闭集候选关系指标和边界错误切片上优于 Random-9，同时 generation-only EM 不下降；结果尚未产生，不能写成已提升。

最低审核证据：

- `environment.txt`：commit、环境、GPU、模型、数据和 adapter hash；
- `metadata.json`：score dump 明确 `split=train`、`filter_splits=["train"]`；
- rollout manifest 与 score dump 的 question ID 完全一致；
- `error_memory.json`：存在 gold-conditioned 错误对；
- `sampler_audit.json`：选中候选的 violation 与 pool 对比；
- `wiring.json`：完整 task 的 relation QA 覆盖和 K=9；
- SwanLab run ID、原始 predictions、run_config、budget 和复现命令。

## 10. 当前不做什么？

- 不启动 GPU rollout、score 或训练；
- 不引入递归执行器或递归深度；
- 不把 `origin/recurrentgrip-nell23k-20260904` 当作本实验父分支；
- 不删除 `origin/main` 中已有的历史快照；它不属于本方法分支新增依赖；
- 不在没有实验结果时声称显著提升、解决 OOV 或提出通用理论。

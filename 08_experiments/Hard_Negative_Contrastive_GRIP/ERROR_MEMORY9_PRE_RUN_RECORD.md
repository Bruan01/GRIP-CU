# ErrorMemory-9 pre-run record

这是当前分支的运行前合同，不是实验结果。生成时间：2026-10-19。

## 版本与拓扑

| 项目 | 当前约定 |
|---|---|
| branch | `caiyang-method` |
| parent | `origin/main` |
| datasets | NELL23K、FB15K237-2 |
| baseline | original GRIP，`13_base_method/grip-exp` |
| method | candidate relation learning + ErrorMemory-9 + InfoNCE |
| recurrence | not used |
| commit | 运行前由 `environment.txt` 写入实际 HEAD |

## 固定协议

- dataset：NELL23K、FB15K237-2；
- train-only error memory and known-fact filtering；
- candidate vocabulary：`train.txt` relations；
- `K=9`；
- `uniform_mix=1/3`；
- `error_weight=1`；
- `violation_weight=1`；
- `sampling_temperature=0.25`；
- `lambda_candidate=1`；
- Random-9 使用相同的原始 GRIP 训练入口和 Stage 2 budget；
- valid/test 只用于评估。

## 资产路径

运行时必须显式确认：

```text
GRIP_BASELINE_DIR=/path/to/GRIP-CU/13_base_method/grip-exp
INPUT_FILE=/path/to/full/grip_nell23k_tasks.json
EVAL_FILE=/path/to/processed_test.json
RAW_DIR=/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/04_benchmark/data/<dataset>/raw
```

`INPUT_FILE` 必须是原始 GRIP task JSON，包含 `context_samples` 和 `qa_samples`；`EVAL_FILE` 必须是标准 graph record，包含 `title`、`graph`、`questions`、`answers`、`id`。FB15K237-2 还必须传入 `RAW_DIR/rel2text.json`；不能再使用递归关系预测文件。

## 分阶段命令

```bash
cd /path/to/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP

GRIP_BASELINE_DIR=/path/to/GRIP-CU/13_base_method/grip-exp \
INPUT_FILE=/path/to/full/grip_nell23k_tasks.json \
EVAL_FILE=/path/to/processed_test.json \
STEP=preflight bash configs/run_error_memory_wsl.sh

STEP=rollout bash configs/run_error_memory_wsl.sh
STEP=score bash configs/run_error_memory_wsl.sh
STEP=freeze bash configs/run_error_memory_wsl.sh
STEP=wire bash configs/run_error_memory_wsl.sh

ALLOW_TRAIN=1 STEP=train \
  SWANLAB_PROJECT=grip-negative-sampling \
  SWANLAB_GROUP=error-memory-k9 \
  bash configs/run_error_memory_wsl.sh
```

第一条只做 read-only preflight。`rollout`、`score` 会加载本地模型并使用 GPU；`freeze`、`wire` 是 CPU；`train` 必须人工审核前面产物后显式授权。

## 运行前检查清单

- [ ] baseline 子模块已经初始化，且 `GRIP_BASELINE_DIR` 正确；
- [ ] 原始 task JSON 是完整文件，不是 smoke/截断文件；
- [ ] `processed_test.json` 是标准 `questions/answers/graph` 格式；
- [ ] `RAW_DIR` 的 `train.txt/valid.txt/test.txt` 与本次数据集版本一致；
- [ ] S1/B1 adapter 和本地模型 cache 已核对 hash；
- [ ] `STEP=preflight` 通过且没有 GPU 工作；
- [ ] rollout/score metadata 都是 `split=train`、`filter_splits=["train"]`；
- [ ] wiring 覆盖完整 task 的 relation QA，并且每条最多/正好 K=9；
- [ ] 训练前保存 branch HEAD、baseline commit、seed、环境、GPU 和复现命令；
- [ ] SwanLab 项目、group、mode 和 run ID 已记录。

## 预期与停止条件

预期是 ErrorMemory-9 在候选关系 MRR/Hits 和边界错误切片上优于 Random-9，同时不降低 generation-only EM。若预期未成立，保留完整失败记录，不把结果解释成“hard negative 必然有效”。任何 train/test 泄漏、ID 不一致、候选数不足或环境/adapter provenance 缺失时，停止训练并先修复证据链。

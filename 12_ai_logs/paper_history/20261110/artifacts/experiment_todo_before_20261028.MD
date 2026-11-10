# 结构反事实对比学习：项目待补充实验 TODO

更新时间：2026-10-23（收窄论文范围，排除 CLEGR）  
对应论文：11_paper/main/main.tex  
方法主线：Structure-Counterfactual Contrastive Learning（SC-CL-9）  
实验状态：NELL23K 已完成 seed=2026 的 SC-CL bank、wiring、训练和闭集测试重排；
WN18RR 与 CLEGR 的本地低预算辅助任务适配器和全测试生成评估已完成；CLEGR 现明确排除在本文范围之外，
不将异机运行或本地低预算结果当作 SC-CL 的迁移、正面或负面证据；
同协议基线、五种子和 FB15K237-2 当前 SC-CL 契约仍未完成；但用户确认的 FB i0 主运行已完成。

## 当前路线决策（2026-10-23）

- 暂沿用现有 SC-CL-9 作为 v1，不立即改算法；论文定位收窄为“开放世界知识图谱关系候选排序的可审计结构反事实对比学习”。
- [x] SCOPE-CLEGR-001：CLEGR-Facts/Reasoning 属于图级组合问答，答案空间、反事实对象和 Judge Accuracy 均不同于 SC-CL；从当前论文与实验矩阵排除。
- [x] 不把未经核验的 CLEGR 异机 +0.415 报告写成论文结果，也不把不适配的 SC-CL 结果表述为“CLEGR 上效果不好”。
- 先在 NELL23K 补齐同预算 B1、Random-9、frequency-matched Random-9，以及 Profile-only/Violation-only 归因对照，并完成 held-out positive overlap 与候选风险审计。
- 审计主要复用现有 bank、逐题候选分数和 test 预测；新对照臂仍需按统一训练契约训练。现有 NELL SC-CL seed=2026 不重跑，继续作为 v1 实测证据。
- 通过 ROUTE-GATE-001 后，再扩展 NELL 五种子和 FB 当前 SC-CL。若频率匹配对照解释了收益，或候选的持出正例冲突风险明显且没有相称收益，则另开风险感知方法分支；该分支需要新 bank 和新训练结果，不能复用 v1 指标冒充新方法结果。
- 所有收益/风险判据须在查看对应 held-out/test 比较结果前登记；valid/test 只用于事后审计，不得参与 bank 构造、阈值调节或选择性调参。

本文档是从论文反推的项目执行清单。这里的“完成”必须意味着有原始 prediction、score、
bank、manifest、配置、环境记录和可复核汇总；不能用 smoke、单元测试或预设论文数字替代。

状态标记：`[x]` = 已完成；`[~]` = 部分完成/仅辅助证据；`[ ]` = 待完成。

## 已完成项目快速清单

- [x] NELL23K：seed=2026 的 train-only SC-CL-9 bank、metadata、sampler audit、wiring 和 hash 已归档。
- [x] NELL23K：SC-CL Stage 2 训练已完成，4,943 条 full test candidate rerank 已完成。
- [x] NELL23K：闭集 MRR/Hits/EM evaluator 已输出，B1 前 200 条配对比较已完成。
- [x] FB15K237-2：用户确认的 InfoNCE-i0 主运行已完成，10,000 条 full candidate score 和分析指标可复核。
- [x] WN18RR：raw/processed/task/关系文本、Qwen-7B 低预算任务适配器和 3,134 条 full test 生成评估已完成。
- [x] CLEGR 本地辅助运行：processed split、低预算 GRIP task、Qwen-7B 任务适配器和 6,460 条 test 生成评估已完成；这不代表外部运行的证据已同步。
- [~] CLEGR 外部运行：用户确认方法已定并在另一台机器完成，报告相对原论文 +0.415；指标尺度、比较行、配置和结果产物待同步，暂不作为已核验结果。
- [x] 工程验证：counterfactual sampler 单元测试 5 passed；有效 NELL 主 run 为 `_06_train`。

以下未勾选项是新增方法对照、provenance 补全或统计扩展，不代表前述实验没有执行。

---

## 0. 当前项目事实与硬边界

| 项目 | 当前状态 | 对实验的影响 |
|---|---|---|
| 运行环境 | /home/kieran/miniconda3/envs/guardenv/bin/python；已知 Python 3.10.19、Torch 2.12.0+cu130 | 所有 GPU、训练、离线 scoring 使用该 Python，禁止切换到 WSL 系统 python3 |
| Python 路径 | 训练入口依赖 baseline 根目录进入 PYTHONPATH | 必须包含 13_base_method/grip-exp、08_experiments/Hard_Negative_Contrastive_GRIP/src 和 scripts |
| baseline | 13_base_method/grip-exp 原始 GRIP | 不修改 baseline；SC-CL 只改变候选关系学习阶段 |
| 当前实现 | counterfactuals.py、build_structure_counterfactual_bank.py、score_relation_candidates_offline.py、listed_training.py 已存在 | 需要从单元级验证升级为全量端到端证据 |
| 本地核心数据 | NELL23K、FB15K237-2 已有 raw、processed 和 GRIP task 文件 | 先完成核心实验；不能用 smoke task 代替完整 task |
| NELL adapter | 13_base_method/grip-exp/artifacts/nell23k_paper_7b/lora_adapter 下存在完整权重文件 | 使用前核对 adapter_config、adapter_model.safetensors、metadata 和 hash |
| FB adapter | 用户确认此前已完成 FB i0 主运行；当前工作区缺少与该运行对应的完整 run record/adapter checkpoint provenance | i0 可作为有效主运行里程碑；在恢复 adapter、环境和配置记录前，不扩展为当前 SC-CL 的 matched-budget 结论 |
| WN18RR | raw、processed、GRIP task、关系文本和 Qwen-7B 低预算任务适配器已存在；全 test 生成评估已完成 | 仅为辅助适配/生成证据；仍需 SC-CL bank、candidate scoring、wiring 和 matched controls |
| YAGO3-10 / CoDEx-S/M | 当前项目未盘点为可运行资产 | 需要下载/固定版本、转换、task 生成、评估和 adapter 接入 |
| CLEGR | 本地低预算 GRIP task/adapter 与 test 评估已完成；用户另报告异机运行较原论文高 +0.415 | 数据集和两类 CLEGR 结果均排除于当前论文；内部保留来源记录，不将其解释为 SC-CL 效果 |
| 0.5B smoke | 只用于验证环境/入口/显存和导入路径 | 不能作为论文主结果、不能与 Qwen2.5-7B 正式结果混合 |

论文中的 NELL 单种子指标来自可复核原始文件；FB i0 是用户确认的有效主运行，候选评分文件
仍可复核，但原始运行记录缺失；WN18RR/CLEGR 本地生成 EM 仅是低预算任务适配器辅助指标，
不是 SC-CL 结果。CLEGR 异机运行的 +0.415 仅作内部来源记录，因任务不在本文范围且原始证据未同步，
其他未运行条目仍是目标格式，不是当前项目已经测得的结果。

## 本轮核对到的新增证据

| 数据/运行 | 已确认产物 | 当前可写结论 | 仍缺的关键证据 |
|---|---|---|---|
| NELL23K SC-CL seed=2026 | `results/runs/..._06_train` 完成；bank/wiring/manifest/adapter/predictions 齐全；4,943 条 full candidate rerank | 主线单种子可复核结果；自由 EM=87.03%，闭集 EM/MRR=91.52%/95.27% | B1 full candidate、Random-9、frequency-matched、消融、五种子 |
| NELL23K 失败尝试 | `_04_train` 因 `assert_error_memory_training_budget` 失败；`_05_train` 因 `TrainingArguments.__init__()` 不接受 `overwrite_output_dir` 失败 | 失败运行只作工程排障记录，不进入主表；`_06_train` 才是有效 run | 不要把失败 run 的日志/目录误认成额外 seed |
| WN18RR auxiliary | raw/processed/task/manifest 已有；任务适配器使用 context=2,048、QA=4,096、Stage 1/2=1/3 epochs；全 test 3,134 条评估完成，EM=43.17%（1,353 条正确） | 数据转换和 Qwen-7B 生成链路可运行；仅标为 auxiliary adapter evaluation | train-only SC-CL bank、B1 scoring、candidate evaluator、wiring、matched controls |
| CLEGR 本地辅助运行 | split 为 300/100/100 graphs；task generation=`complete_low_budget` 且 train/reference 均为 train；全 test 6,460 条，EM=27.15%（1,754 条正确） | 仅为本地低预算任务适配记录，不进入本文，不作为 SC-CL 证据 | 无当前论文后续任务 |
| CLEGR 异机运行（用户报告） | 用户确认方法已定且异机运行完成；报告比原论文高 +0.415 | 仅留档；方法、指标和原始产物未核验，且不属于当前论文范围 | 不列入当前论文追补项；若未来重启 CLEGR 方向，再单独立项并核验 |
| FB15K237-2 i0 主运行 | 用户确认此前已完成；10,000 条 B1/i0 full candidate score、分析指标和数据 hash 可复核 | 作为有效 FB 主运行里程碑记录；方法名称保留为 `InfoNCE-i0`，不擅自改名为 SC-CL | 原始 run_config、environment、adapter/checkpoint hash、命令和当前 SC-CL bank；B1/Random-9/SC-CL-9 matched-budget 对照 |

---

## 1. 优先级总览

### P0：主实验闭环与结果复核阻塞项

- [x] ENV-001（运行环境已核验）：guardenv、CUDA、模型缓存、baseline 导入和 PYTHONPATH 已在 NELL、FB、WN18RR、CLEGR 的实际运行中确认。
- [ ] ENV-001-REMAIN：补回 FB i0 缺失的 environment/adapter provenance，并保存一次统一 preflight 日志。
- [x] DATA-001-PATH（核心资产已核验）：NELL23K/FB15K237-2 的 raw、task、eval、关系文本路径已核对并用于实际运行。
- [ ] DATA-001-QA：补齐两数据集的实体/关系/三元组/QA 计数、alias 覆盖、重复 ID、adapter/raw/task 统一 hash 表。
- [x] TEST-001-UNIT：counterfactual sampler 单元测试 5 passed。
- [ ] TEST-001-REMAIN：运行全量项目测试，并补齐 score → freeze → wire 的 CPU 集成测试。
- [ ] RUNNER-001：补齐 Random-9 的可执行 runner；当前 shell runner 只接受 error_memory 和 structure_counterfactual。
- [x] BANK-001：NELL23K 训练 split 的冻结 B1 score、SC-CL bank、metadata、sampler audit 和 wiring（run seed=2026 已归档）。
- [x] MAIN-NELL-001-SC：NELL23K SC-CL seed=2026 已完成训练、full candidate rerank 和闭集/自由生成评估。
- [ ] MAIN-NELL-001-REMAIN：先补齐 seed=2026 的 B1 full candidate、Random-9、frequency-matched 和同预算三臂主比较，再通过 ROUTE-GATE-001 决定是否扩展种子。
- [x] EVAL-001：实现测试集闭集候选评估，输出 MRR、Hits@1/3/10、wrong-in-list 和 margin（NELL full 4,943 条已完成）。
- [ ] SEED-001：NELL23K 五个训练种子 2026--2030；在 seed=2026 公平对照与风险审计通过路线门后启动，复用固定 bank 和完全相同的 test question IDs。

### P1：论文主张成立所需的机制、安全和统计证据

- [x] MAIN-FB-001-I0（主运行已确认）：FB15K237-2 的 InfoNCE-i0 主运行已由用户确认完成，10,000 条 full candidate score 可复核。
- [ ] MAIN-FB-001-REMAIN：恢复 i0 的 run_config/environment/adapter/checkpoint provenance，并补齐当前 SC-CL 契约、B1/Random-9/SC-CL matched-budget 对照。
- [ ] ABL-001：结构信号、B1 violation、均匀下限、InfoNCE、K 和过滤消融。
- [ ] CTRL-001：frequency-matched Random-9，排除关系频率捷径。
- [x] AUDIT-001-NELL-WIRING：NELL train-only、question-ID、candidate 去重、bank 泄漏和 score→freeze→wire 产物审计已完成。
- [ ] AUDIT-001-REMAIN：补齐 NELL held-out/OWA 审计和 FB 当前 SC-CL bank 审计。
- [x] ROUTE-DECISION-001：确定先沿用 SC-CL v1 完成公平对照与风险审计；风险感知方法仅在预登记路线门触发后另开分支。
- [ ] ROUTE-GATE-001：依据预登记的配对收益与持出正例冲突判据，决定继续扩展 SC-CL v1 或启动风险感知分支。
- [ ] SLICE-001：一对多、多对一、关系频率、图度、画像稀疏和 B1 高置信混淆切片。
- [ ] STAT-001：均值/标准差、paired bootstrap 95% CI、逐题差异和多重比较校正。
- [x] COST-001-NELL（已完成部分）：NELL bank 构造、B1 scoring、训练、推理 wall-clock、显存和 token 已记录。
- [ ] COST-001-REMAIN：补齐 CPU 峰值、完整 GPU 曲线和跨数据集可比成本。

### P2：跨数据集和外部定位

- [x] DATA-WN-001（任务适配阶段）：WN18RR raw/processed/task/关系文本、低预算 Qwen-7B 适配器和 3,134 条 full test 生成评估已完成；SC-CL bank、candidate scoring、wiring 和 matched controls 待完成。
- [ ] DATA-FB-001：canonical FB15K-237 与本地 FB15K237-2 分开接入，不重复计权。
- [ ] DATA-YAGO-001：YAGO3-10 数据和 GRIP task 适配。
- [ ] DATA-CODEX-001：CoDEx-S/M verified-negative 审计协议。
- [ ] DATA-CODEX-002：CoDEx-L 可选大规模压力测试。
- [x] SCOPE-CLEGR-001：CLEGR-Facts/Reasoning 从当前论文和 SC-CL 实验矩阵排除；本地已有结果只保留数据资产记录。
- [x] SCOPE-CLEGR-002：异机 +0.415 只记入内部状态，不补写当前主文，不称为 SC-CL 的好或坏结果。
- [ ] EXT-001：RotatE、ComplEx、ConvE、SimKGC 或 SANS-style 外部参考；只在统一协议后报告。

---

## 2. P0：环境、数据和代码入口

### ENV-001：固定实验环境

- [x] 确认正式运行使用 `/home/kieran/miniconda3/envs/guardenv/bin/python`、CUDA GPU 和项目 PYTHONPATH；NELL/FB/WN18RR/CLEGR 运行 metadata 均记录该环境。

~~~bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU
export PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/13_base_method/grip-exp:/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/src:/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/scripts
$PYTHON -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
PYTHONPATH=$PYTHONPATH $PYTHON -c "import grip; import hard_negative_grip; print('imports-ok')"
~~~

- [x] 记录 NELL 正式 run 的 WSL/CUDA/GPU/Torch/模型缓存和环境文件；WN18RR/CLEGR eval metadata 也记录 Python、模型和输入/输出 hash。
- [x] 确认 `/mnt/e/GRIP-CU/model_cache/Qwen--Qwen2.5-7B-Instruct` 可加载；WN18RR/CLEGR full evaluator 已实际加载并完成测试。
- [~] NELL 正式 run 已写 `environment.txt` 和 hash；FB i0 的原始 environment/adapter 记录缺失，需恢复 provenance。
- [ ] 0.5B smoke 只记录为 auxiliary/dev；即使通过，也不能替代 7B full-task 证据。

推荐的只读项目 preflight：

~~~bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
export PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
export PYTHONPATH=/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/13_base_method/grip-exp:/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/src:/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/scripts
DATASET=nell23k STEP=preflight PYTHON=$PYTHON bash configs/run_structure_counterfactual_wsl.sh
~~~

### DATA-001：核心数据资产审计

- [x] NELL23K 核对：
  - 04_benchmark/data/nell23k/raw/train.txt、valid.txt、test.txt；
  - 04_benchmark/data/nell23k/tasks/grip_nell23k_tasks.json；
  - 04_benchmark/data/nell23k/processed/processed_test.json；
  - raw/entity2text.json；
  - task JSON 必须包含 context_samples、qa_samples；
  - processed test 必须包含 graph、questions、answers、id。
- [x] FB15K237-2 核对：
  - raw/train.txt、valid.txt、test.txt；
  - raw/rel2text.json；
  - tasks/grip_fb15k237_2_tasks.json；
  - processed/processed_test.json；
  - 关系文本别名必须能映射回 train vocabulary 的精确 label。
- [~] 计算并记录每个数据集的实体数、关系数、train/valid/test 三元组数、QA 总数、
  relation QA 数、可匹配 QA 数、重复 question ID 数和 relation text 覆盖率。
- [~] 核对当前 runner 的 expected listed 数：NELL23K 3253 已通过；FB15K237-2 的 3109 仍需在当前 SC-CL runner 中复核；
  若实际盘点不同，先查 task 版本和 relation alias，不得直接改 expected 值掩盖问题。
- [~] NELL raw/task/processed/adapter、WN18RR raw/processed 和 FB i0 数据文件已有 hash；FB 原始 run/adapter hash 仍缺失。
- [x] 已区分完整 test evaluator、低预算 adapter 和 smoke/pilot；NELL SC-CL 主 run 通过 full-task budget gate。

### TEST-001：从单元测试升级到集成测试

- [ ] 使用 guardenv 运行整个 HNG 测试目录，而不只运行 test_counterfactuals.py：

~~~bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
/home/kieran/miniconda3/envs/guardenv/bin/python -m pytest -q
~~~

- [x] 保留当前 counterfactual sampler 的 5 个通过测试作为实现证据，但没有将其当作模型结果。
- [ ] 新增/补齐 CPU 集成 fixture：候选 score JSONL → SC bank → wiring.json → task attachment。
- [ ] 集成 fixture 必须覆盖：
  - 非 train score group 被拒绝；
  - train-known relation 不进入候选；
  - 候选恰好 K 个且互不重复；
  - original question ID 保持 task_qa:N；
  - FB relation alias 可以 canonicalize；
  - bank 已存在时默认拒绝隐式覆盖；
  - rows_failed 被记录而不是静默丢弃。
- [ ] 运行 shell 的静态检查和参数检查；任何 runner 能接受但训练脚本不识别的参数都要在 GPU 前修复。

---

## 3. P0：冻结 B1 和 SC-CL bank

### BANK-001：NELL23K 主 bank

目标：只用 NELL23K train split 生成一次冻结的 SC-CL-9 bank，固定 bank seed=2026，
供所有训练 seed 复用。B1 scoring、结构画像、候选采样和 wiring 不能读取 valid/test。

- [x] 确认 B1 adapter 是可加载的 frozen generation-only adapter，并在 NELL run environment/manifest 中记录路径和 hash。
- [x] 使用独立 ARTIFACT_ROOT，未覆盖历史 ErrorMemory 或其他数据集的 bank。
- [x] 执行 score：

~~~bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP
export PYTHON=/home/kieran/miniconda3/envs/guardenv/bin/python
export DATASET=nell23k
export ARTIFACT_ROOT=/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/sccl_bank_nell23k_seed2026
export S1_ADAPTER=/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/13_base_method/grip-exp/artifacts/nell23k_paper_7b/lora_adapter
export B1_ADAPTER=$S1_ADAPTER
export MODEL_CACHE_DIR=/mnt/e/GRIP-CU/model_cache
STEP=score NEGATIVE_SOURCE=structure_counterfactual PYTHON=$PYTHON bash configs/run_structure_counterfactual_wsl.sh
~~~

- [x] 检查 b1_train_scores/candidate_scores.jsonl、qa_summary.jsonl、run_stats.json、
  metadata.json 的完整性，记录 matchable QA 数、relation vocabulary size、skipped 数、
  total candidates、elapsed_sec、peak_gpu_gib。
- [x] 执行 freeze，默认参数固定为 K=9、rho=1/3、alpha=1、beta=1、tau_s=0.25：

~~~bash
STEP=freeze NEGATIVE_SOURCE=structure_counterfactual PYTHON=$PYTHON bash configs/run_structure_counterfactual_wsl.sh
~~~

- [x] 执行 wire，EXPECTED_LISTED=3253；输出 wiring.json：

~~~bash
EXPECTED_LISTED=3253 STEP=wire NEGATIVE_SOURCE=structure_counterfactual PYTHON=$PYTHON bash configs/run_structure_counterfactual_wsl.sh
~~~

- [x] bank 产物必须存在：
  - counterfactual_bank.jsonl；
  - sampler_audit.jsonl；
  - relation_profiles.json；
  - metadata.json；
  - wiring.json；
  - score metadata 和原始 candidate_scores.jsonl。
- [x] 统计 rows_written、rows_failed、每个 pool size、selected/pool structure similarity、
  selected/pool violation、candidate entropy 和 failure 原因。
- [x] 对 bank JSONL 做独立审计：split=train、filter_splits=[train]、K=9、无重复、
  无 gold、无 train-known、question ID 唯一且与完整 task 对齐。
- [x] 记录 bank hash；正式五种子训练不得修改该 bank。

### BANK-002：FB15K237-2 bank

- [ ] 先完成 FB adapter 资产恢复或重新训练；没有完整 adapter_model.safetensors 时停止。
- [ ] 运行所有 score/freeze/wire 命令时显式提供：
  - DATASET=fb15k237_2；
  - RELATION_TEXT_FILE=/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/04_benchmark/data/fb15k237_2/raw/rel2text.json；
  - EXPECTED_LISTED=3109；
  - 独立 ARTIFACT_ROOT；
  - S1/B1 adapter hash。
- [ ] 不把 FB15K237-2 的 bank 与 canonical FB15K-237 混称。

---

## 4. P0：主比较和公平性

### RUNNER-001：补齐 Random-9 执行入口

当前 train_listed_contrastive.py 支持 listed_negative_source=train_graph、
negative_sampling_mode=random，但 run_error_memory_wsl.sh 的 NEGATIVE_SOURCE 校验目前
只接受 error_memory 和 structure_counterfactual；random9_pair.json 因此不是完整可执行
主控协议。

- [ ] 二选一并固定：
  1. 扩展 run_error_memory_wsl.sh，使 NEGATIVE_SOURCE=train_graph 走同一个 preflight、
     train、evaluate、environment 和 comparison 输出；
  2. 写一个等价的随机控制 runner，显式传入 listed_negative_source=train_graph、
     negative_sampling_mode=random、listed_negative_k=9、listed_negative_seed=2026。
- [ ] 不能只手工调用训练脚本而丢失 run_config、wiring、budget、environment 和日志。
- [ ] Random-9 必须使用：
  - 相同完整 task；
  - 相同 train relation vocabulary；
  - 相同 train-only known-pair filter；
  - 相同 K=9；
  - 相同 S1 adapter；
  - 相同 Stage 2 epochs、batch、gradient accumulation、LoRA、lambda 和温度；
  - 固定 bank seed/negative seed；
  - 相同五个 training seeds。
- [ ] 增加 frequency-matched Random-9 前，先保证普通 Random-9 已可独立复现。

### MAIN-NELL-001：NELL23K 一次完整主实验

当前已完成 SC-CL seed=2026 的正式训练、冻结 bank、wiring 和 4,943 条 full candidate
rerank；本阶段先补同 seed、同预算的 B1 full candidate、Random-9、频率匹配和核心机制对照，
再按 ROUTE-GATE-001 决定是否投入完整多种子与跨数据集确认实验。单种子筛查只用于路线/资源决策，
不能作为最终统计结论。

- [ ] 每个 seed 使用新的 RUN_ID/RUN_DIR，禁止覆盖已有目录。
- [ ] 建议的主 arm：
  - Original GRIP/B1：lambda_candidate=0，generation-only；
  - Random-9：train_graph + InfoNCE；
  - SC-CL-9：frozen structure_counterfactual bank + InfoNCE；
  - ErrorMemory-9：只作为辅助 ablation，不作为主创新。
- [x] seed=2026 的 SC-CL 主路径已完成并归档；B1 full candidate、Random-9 和 2027--2030
  仍待运行。
- [ ] 训练配置必须在 run_config.json 中显式记录：
  - qwen-7b；
  - LoRA r=4、alpha=8、down_proj/up_proj/gate_proj；
  - Stage 2 配置和 involve_qa_epochs；
  - per_device_train_batch_size=1；
  - gradient_accumulation_steps=512；
  - lambda_candidate=1.0；
  - candidate temperature=1.0；
  - generation max length=32；
  - K=9、rho=1/3、alpha=1、beta=1、tau_s=0.25；
  - dataset、task/eval/raw 路径和全部 hash。
- [ ] 输出必须至少包括：
  - run_config.json；
  - environment.txt；
  - budget.json；
  - wiring.json/wiring.log；
  - b1/adapter、b1/predictions_correct.jsonl、b1/summary.json；
  - listed/adapter、listed/predictions_correct.jsonl、listed/summary.json；
  - comparison.json；
  - trainer 日志、checkpoint 和 SwanLab offline/online run ID（若启用）。
- [ ] 不使用历史 RCCGRIP、bridge、novel2hop 或 0.5B 的指标作为本主比较结果。

### MAIN-FB-001：FB15K237-2 主实验

- [x] 用户确认此前已完成 FB15K237-2 `InfoNCE-i0` 主运行；已有 10,000 条 full candidate score、B1/i0 分析和数据 hash，可作为有效主运行里程碑。
- [ ] 补回该运行的 run_config、environment、adapter/checkpoint hash、命令和完整原始预测 provenance；缺失记录不等于删除该主运行，但需要在论文主表前补齐。
- [ ] 若要作为当前 SC-CL 主结果，仍需 DATA-001、BANK-002、train-only SC-CL bank、wiring 和 matched-budget B1/Random-9/SC-CL-9 对照。
- [ ] NELL23K v1 通过 ROUTE-GATE-001 后，用相同 SC-CL 三臂和统一训练契约在 FB15K237-2 重新核验；不能把 i0 的不同 LoRA/训练契约直接改名为 SC-CL。
- [ ] 记录 relation text alias coverage、非 relation QA、unmatched QA 和 relation QA 数。

---

## 5. P0/P1：闭集候选评估、自由生成和指标

### EVAL-001：实现测试集闭集候选评估

NELL23K 的 SC-CL full evaluator 已输出 4,943 条逐题候选分数和 summary；FB i0 的 full
candidate evaluator 也已有输出。以下清单保留用于复核字段、B1/Random-9 对照和 FB 当前
SC-CL 的缺口。

当前 NELL SC-CL 已有独立、可复核的测试集闭集 MRR/Hits 输出；以下任务用于补齐 B1、
Random-9、FB 当前 SC-CL 的同协议输出，并检查所有论文字段。

- [x] 明确并冻结两种 candidate evaluation protocol：
  - 训练 bank diagnostic：对 train QA 使用冻结 bank 计算候选 score，只用于检查训练/机制；
  - test closed-set primary：候选集合的构造不能读取 valid/test 三元组或测试标签；
    gold 只在最后用于计算 rank，不得参与候选抽样。
- [x] NELL test closed-set primary 使用 train vocabulary、train graph profile 和冻结 B1
  query score 生成 label-blind candidate set；如果实现 gold-conditioned test set，
  必须单独标成 diagnostic，不得作为部署安全的主证据。
- [x] 输出逐题 candidate_predictions.jsonl，至少包含：
  - question_id；
  - gold_relation；
  - candidate_relations；
  - positive_score；
  - negative_scores；
  - positive_rank；
  - reciprocal_rank；
  - hits_at_1/3/10；
  - hardest_negative；
  - positive_minus_hardest_margin；
  - candidate_source；
  - filter_splits；
  - bank/evaluator hash。
- [x] 输出 candidate_summary.json，至少包含：
  - count；
  - MRR；
  - Hits@1、Hits@3、Hits@10；
  - wrong-in-list；
  - candidate coverage；
  - valid candidate rate；
  - mean margin；
  - score tie count；
  - unresolved/invalid candidate count。
- [~] 扩展自由生成 summary：NELL EM、candidate_valid_rate、wrong-in-list/wrong-out-of-list 已有；OOV/NLL 统一字段仍待补。
  - EM；
  - OOV：规范化答案不在 train relation vocabulary 的比例；
  - wrong-in-list；
  - wrong-out-of-list；
  - generation NLL 或可比较的生成 loss；
  - candidate_valid_rate。
- [x] 候选指标和自由生成指标分开报告，没有用 EM 代替 MRR，也没有用 train bank 指标代替 test。
- [~] evaluator 的主要 rank/metric 规则已固定；tie 的保守最坏 rank 仍需在统一 metrics.py 中补一份独立验证。

### METRIC-001：统计与逐题对齐

- [ ] 所有 arm 使用完全相同的 test question IDs，并保存 question ID 对齐报告。
- [ ] 生成每题的 B1、Random-9、SC-CL-9 paired rows，禁止只保留平均值。
- [ ] 增加一个 CPU 聚合脚本，输入原始 JSONL，输出：
  - per_seed_summary.json；
  - paired_effects.jsonl；
  - aggregate.json；
  - missing_or_duplicate_ids.json。
- [ ] 统一百分比和小数单位；内部计算用 0--1，论文汇总再乘 100。
- [ ] 复核 MRR、Hits 的 rank 方向、score 高低方向、answer normalize 和 relation alias。

---

## 6. P1：机制消融、控制和频率捷径

### ABL-001：SC-CL 机制消融

先在 NELL23K seed=2026 做开发验证，再在主结论需要时扩展五种子。每个变体都必须保留
相同 task、S1 adapter、Stage 2 budget、test evaluator 和审计输出。

- [ ] ABL-PROFILE：structure_weight=0，保留 B1 violation。
- [ ] ABL-VIOLATION：violation_weight=0，保留 structure profile。
- [ ] ABL-UNIFORM：uniform_mix=0，检查候选熵、有效率和 MRR。
- [ ] ABL-INFONCE：lambda_candidate=0，作为 generation-only B1，不把它称为 SC-CL。
- [ ] ABL-K3：K=3；ABL-K15：K=15。
- [ ] ABL-TAU：tau_s=0.10、0.25、0.50。
- [ ] ABL-RHO：rho=0、1/3、2/3。
- [ ] ABL-FILTER-RISK：关闭 train-known filter 只做风险诊断，不进入主方法结论；
  输出冲突率，不得因为 MRR 更高而把它当成合法方法。
- [ ] 当前实现/verify 代码固定 K=9 的位置需要参数化；K 敏感性必须通过 expected_k、
  manifest requested_k/actual_k 和 evaluator 同步校验，不能只改配置字符串。

### CTRL-001：Frequency-matched Random-9

- [ ] 从 train graph 统计每个关系频率，按分位数或预注册频率桶采样。
- [ ] 对每个问题，让随机候选的频率桶分布匹配 SC-CL 候选，而不是事后挑选最优随机种子。
- [ ] 输出 frequency_matched_random_bank.jsonl、metadata、audit 和 bank hash。
- [ ] 用相同 K=9、train-only filter、question IDs、seed 和训练预算。
- [ ] 若 Frequency-matched Random-9 与 SC-CL 相当，论文只能声称频率匹配后的额外收益不足。

### CTRL-002：ErrorMemory-9 辅助对照

- [ ] 在 NELL23K 至少运行 seed=2026；若其输出稳定且资源允许，再运行五种子。
- [ ] 保持 ErrorMemory 为 legacy auxiliary，不把它写成 SC-CL 主线。
- [ ] 同时报告其 candidate MRR/Hits、EM、OOV、bank audit 和成本。

---

## 7. P1：假阴性、泄漏和失败切片

### AUDIT-001：train-only 和 known-pair 硬门

- [ ] 新增或补充独立 audit 脚本，不只依赖训练入口内部检查。
- [ ] 对每条 bank row 检查：
  - split 必须为 train；
  - filter_splits 只能为 train；
  - gold 不在 negative_relations；
  - candidate relations 属于 train vocabulary；
  - candidate relations 不属于当前实体对 train-known relations；
  - candidate 数量恰为 K；
  - candidate 无重复；
  - question ID 与 task 一一对齐；
  - bank hash、score metadata、配置和 sampler 版本可追溯。
- [ ] 硬门任何一项非零都作废该运行，不能只在论文脚注解释。
- [ ] 检查 valid/test 是否以任何形式进入候选频率、B1 score、profile、filter 或采样 seed。

### AUDIT-002：持出正例和开放世界审计

- [ ] train-known collision 必须报告为 0。
- [ ] 用 valid.txt/test.txt 只做事后诊断，统计候选与持出正例的 overlap；
  valid/test 不得回流 bank。
- [ ] 分开报告：
  - train-known conflict；
  - valid/test positive overlap；
  - external/verified positive overlap；
  - unresolved open-world candidate。
- [ ] unresolved OWA 不能计作“真实负例错误率”。
- [ ] CoDEx-S/M 进入后，使用其 verified-negative 文件报告 purity 和 positive collision；
  不要把 verified negatives 静默并入训练 graph。

### ROUTE-GATE-001：SC-CL v1 继续验证或启动风险感知分支

本门用于决定后续算力投入，不替代多种子统计，也不允许根据 test 结果临时改方法。

- [ ] 在运行/查看 held-out 比较前，登记最小有意义的配对排序收益阈值 `delta_MRR`，以及相对 Frequency-matched Random-9 可接受的额外持出正例冲突上限 `delta_overlap`；说明阈值依据并冻结到实验计划。
- [ ] 以 NELL23K seed=2026 的同预算 B1、Random-9、Frequency-matched Random-9、Profile-only、Violation-only 为初筛；同时报告自由生成 EM/OOV 和闭集 MRR/Hits，保持逐题 ID 配对。
- [ ] 只把 valid/test positive overlap 当作 bank 的事后风险诊断；该信息不进入候选、采样、阈值拟合或训练。
- [ ] 若 SC-CL 相对频率匹配随机达到预登记的实际收益门槛，且额外持出正例冲突未超过预登记风险上限，则保留 v1，继续 NELL 五种子和 FB 当前协议验证。
- [ ] 若频率匹配随机已解释主要收益，或持出正例冲突明显超出风险上限且排序收益不足，则暂停 v1 的强方法主张，另开“风险感知候选”方法分支；先定义软标签/风险权重和独立审计，再构造新 bank、训练新模型。
- [ ] 若单种子结果落在不确定区间，不据此宣布有效或无效；结合算力预算决定完成预注册的多种子确认，或将结论收窄为探索性结果。
- [ ] 无论初筛结果如何，单 seed 结果都不能支持稳定性或显著性结论；通过初筛后仍须完成 SEED-001、配对区间和第二数据集验证。

### SLICE-001：失败分析

- [ ] 一对多、 多对一、 一对一关系切片。
- [ ] 头/尾实体度或一跳邻域度分位数切片。
- [ ] 关系频率分位数切片。
- [ ] relation profile 稀疏度切片。
- [ ] B1 高置信混淆与低置信样本切片。
- [ ] 每个切片报告 n、B1、Random-9、SC-CL-9 的 EM、MRR、H@10、margin。
- [ ] n<100 的切片只做描述性统计，不进行显著性结论。
- [ ] 输出失败样本 question ID 列表和可复查的 candidate score，不只输出计数。

---

## 8. P1：五种子、置信区间和资源

### SEED-001 / STAT-001：正式统计协议

- [ ] 主训练种子固定为 2026、2027、2028、2029、2030。
- [ ] bank seed 固定为 2026；所有 arm 复用同一 frozen bank 版本，以隔离 sampler 和训练 seed。
- [ ] B1 teacher、S1 adapter、task、eval、candidate evaluator 和 bank hash 在同一数据集的五种子间保持一致；
  若选择每 seed 重建 teacher/bank，必须作为另一实验轴单独登记。
- [ ] 计算：
  - mean ± std；
  - SC-CL minus Random-9 的 paired per-question effect；
  - 10,000 次 paired bootstrap 95% CI；
  - reciprocal-rank 差异的配对检验；
  - 多切片/多超参族的 Benjamini--Hochberg 校正。
- [ ] 预设论文表不填写 p 值和星号；只有原始预测和脚本齐备后才写显著性。
- [ ] 若不同 seed 效应方向不一致，结论标为 inconclusive，不挑选有利 seed。

### COST-001：资源—收益

NELL seed=2026 已记录 B1 scoring、bank 行数、Stage 2 token/时间和测试重排时间；仍需
补独立 CPU 峰值、完整 GPU 曲线及 FB/迁移数据集的可比记录。

- [ ] 记录关系 profile CPU 时间、B1 full-vocabulary scoring GPU 时间、bank freeze 时间、
  wiring 时间、Stage 2 训练时间、test inference 时间。
- [ ] 记录 bank 行数、候选总数、CPU 峰值内存、GPU 峰值显存、训练 token、checkpoint 大小。
- [ ] score_relation_candidates_offline.py 的 benchmark.json 和 run_stats.json 纳入产物。
- [ ] 训练日志中区分 generation_loss、candidate_loss、contrastive_candidate_loss、
  candidate_forwards 和最后 checkpoint。
- [ ] 输出每 1M candidate 的 bank/scoring 成本和每 GPU-hour 的 paired MRR gain；
  没有真实计时就不写资源数字。

---

## 9. P2：数据集扩展

### DATA-WN-001：WN18RR

- [x] 盘点 04_benchmark/data/wn18rr/raw、processed、wn18rr.json 和 manifest；raw/processed 哈希已写入 manifest。
- [x] 从 raw/processed 生成原始 GRIP task JSON，保留 train/valid/test split 和 question ID；generation plan 标记 complete。
- [x] 解决 WN18RR relation text：`raw/wn18rr.json` 已固定关系命名/方向说明。
- [~] Qwen-7B task adapter 与 full test generation evaluator 已跑通；训练仍是低预算 context=2,048、QA=4,096、Stage 1/2=1/3 epochs。
- [ ] 扩展/确认 SC-CL runner 的 DATASET 分支，生成 train-only B1 score、bank、wiring 和 candidate metrics。
- [ ] 完成 WN18RR 的 relation QA 数、expected listed、adapter hash、score、bank、wire 和 matched evaluator 审计。
- [ ] 在 SC-CL 协议下先做 seed=2026 auxiliary，再决定是否纳入五种子主结果。

### DATA-FB-001：canonical FB15K-237

- [ ] 获取并固定 canonical FB15K-237 的 raw 文件、版本和 hash。
- [ ] 与本地 FB15K237-2 分别保存数据根目录、relation text、task JSON、adapter 和 evaluator 输出。
- [ ] 论文中不把两者合并计数，也不把相同图规模当作两个独立泛化证据。
- [ ] 只有完整 task、relation alias、adapter 和 train-only bank 通过后才进入扩展表。

### DATA-YAGO-001：YAGO3-10

- [ ] 固定公开下载版本和 parser；记录实体/关系去重规则。
- [ ] 生成 relation QA、图上下文和标准 processed graph record。
- [ ] 解决大规模 task 的内存、候选 scoring 分片、bank 行数和资源上限。
- [ ] 运行前先做 read-only inventory 和 100--500 QA adapter smoke；smoke 不写论文结果。
- [ ] full run 至少报告 MRR/Hits、EM、bank/scoring/training 成本。

### DATA-CODEX-001 / DATA-CODEX-002：CoDEx-S/M/L

- [ ] 固定 CoDEx-S/M（必要时 L）的官方版本、正例、verified-negative 文件和 hash。
- [ ] 转换为原始 GRIP task 时保留正例和 verified-negative 的独立字段。
- [ ] 候选训练仍只使用 train split；verified-negative 只用于 audit/evaluation。
- [ ] 输出 verified-negative purity、positive collision、unresolved OWA、MRR/Hits。
- [ ] CoDEx-L 为可选资源压力测，不得用它替代 NELL/FB 核心证据。

### SCOPE-CLEGR-001：CLEGR 不纳入当前论文

- [x] 本地 `clegr_facts`/`clegr_reasoning` 资产继续保留在 benchmark 仓库，供其他研究使用；不纳入本文的 dataset list、主表和结果主张。
- [x] 当前 SC-CL 不直接应用于 CLEGR：关系标签替换无法覆盖 CLEGR 的图编辑、答案重算和类型化答案判别。
- [x] 外部 +0.415 作为用户报告留档；不据此写“效果好/不好”，不要求为当前论文补交或重跑该实验。
- [ ] 若未来另立 CLEGR 研究，需新建独立方法与评估协议；最低门槛为经验证的逐类型答案程序、合法图编辑、答案变化且唯一、严格按 GRIP Judge Accuracy 评估，并完成图级 split 与 question-ID 泄漏审计。

---

## 10. P2：外部基线和定位

### EXT-001：外部参考

- [ ] RotatE、ComplEx、ConvE、SimKGC 只在统一 filtered link-prediction 或候选协议下运行。
- [ ] SANS-style structural control 只有在结构定义、负例过滤和预算可比时才加入。
- [ ] 外部方法不与 GRIP 的生成式 EM/OOV 直接混排，不用不一致的候选集合宣称全面优于。
- [ ] 外部基线不是 SC-CL 核心主张的 P0 阻塞项；核心证据不足时先不要扩张外部比较。

---

## 11. 统一运行产物要求

每一个正式 run 目录都必须至少有：

- run_config.json；
- environment.txt；
- command.sh 或完整可复制命令；
- stdout/stderr 日志；
- data/adapter/model/baseline hash；
- budget.json；
- wiring.json；
- bank metadata、manifest、audit 和 hash；
- B1/listed adapter 或明确的 frozen adapter 来源；
- free-generation predictions 和 summary；
- closed-set candidate predictions 和 summary；
- per-question paired metrics；
- resource.json 或等价资源记录；
- failure list；
- README：运行目的、状态、结果、限制和下一步。

建议目录：

~~~text
08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/
  sccl_bank_nell23k_seed2026/
  sccl_nell23k_seed2026/
  random9_nell23k_seed2026/
  b1_nell23k_seed2026/
  sccl_nell23k_seed2027/
  ...
~~~

---

## 12. 论文写回规则

- [ ] 对每个方法/数据集/seed，只有原始预测、candidate score、bank manifest、wiring、config
  和 hash 齐备后，才能标为 `observed`；单个 seed 可以标为 `observed-single-seed`，但不得
  删除稳定性和对照缺口。
- [ ] 更新 11_paper/results_status.md、paper_experiment_matrix.md/json、artifacts/evidence_ledger.md、
  review/review.md、review/experiment_todo.md 和 revision_log.md。
- [ ] 每次补齐一批直接对照后重新生成 MRR 图、跨数据集 delta 图和所有表；当前已生成
  `11_paper/artifacts/SC-CL_MRR_observed.png`，覆盖 NELL 配对结果和 FB i0 主运行辅助面板；
  后续若补齐当前 SC-CL 结果再重绘主图。
- [ ] 对每个论文结论标注 supported、refuted 或 inconclusive。
- [ ] 如果 SC-CL 不优于 frequency-matched Random-9，结论改为“当前设置下未观察到额外收益”。
- [ ] 如果闭集 MRR 上升但 EM/OOV 恶化，结论只能写成候选排序改善，不能写成生成能力改善。
- [ ] 如果 valid/test 泄漏、known-pair 冲突、question ID 不对齐或 K 不一致，整次 run 不进入主表。

---

## 13. 推荐执行顺序

1. ENV-001 → DATA-001 → TEST-001；
2. RUNNER-001，确保 Random-9 能和 SC-CL 使用同一运行契约；
3. 复用已完成的 NELL seed=2026 SC-CL、BANK-001 和 EVAL-001；补同协议 B1 full candidate 与 Random-9；
4. 在 NELL seed=2026 补 Frequency-matched Random-9、Profile-only、Violation-only，并对所有候选库完成 held-out/OWA 风险审计；
5. 在查看对应 held-out/test 比较前，冻结 ROUTE-GATE-001 的收益阈值、风险上限和判定规则；完成单种子初筛后只作算力/路线决策，不作稳定性结论；
6. 若 v1 通过路线门，执行 NELL 的 2027--2030 种子和 FB 当前 SC-CL 的 BANK-002/同预算对照；若触发风险分支，则先定义、审计并实现新方法，再为新方法生成独立 bank 和训练结果；
7. 对最终保留的路线补全 ABL-001、AUDIT-001/002、SLICE-001、METRIC-001 和 STAT-001；若初筛结果不确定，不能将其写成支持或否定方法的结论；
8. 已确认 FB i0 主运行并保留其 InfoNCE-i0 标签与主运行记录；按统一协议补齐 provenance，作为辅助历史证据，不替代 FB 当前 SC-CL；
9. WN18RR 的任务适配已完成但仍需 SC-CL bank/controls；核心 NELL/FB 结论稳定后，再接入 canonical FB、YAGO3-10、CoDEx-S/M。CLEGR 不纳入当前论文；
10. 根据真实且可追溯的结果更新统计汇总、证据账本和论文；不将预设数字、单种子筛查或旧方法结果写成当前方法结果。

---

## 14. 明确不做

- [ ] 不把 0.5B smoke 当成论文结果。
- [ ] 不用 smoke task、64-QA 子集或 underfit bypass 生成主结果。
- [ ] 不把 valid/test 三元组用于 bank、profile、frequency、score 或 filter。
- [ ] 不把 train 中未观测关系称为真实负例。
- [ ] 不把 FB15K237-2 与 canonical FB15K-237 重复计权。
- [ ] 不把 CLEGR 的用户报告 +0.415 或本地低预算 EM 移入当前论文，也不以不匹配的关系候选方法得出 CLEGR 负结果。FB i0 保留 InfoNCE-i0 方法标签和 provenance 缺口。
- [ ] 不在没有原始证据时填入显著性、置信区间、资源耗时或跨数据集结论。

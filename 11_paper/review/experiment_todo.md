# Formal experiment TODO

所有任务完成后才可以把正文中的 preset/planned 改成 observed。任务 ID 与
paper_experiment_matrix.md、paper_experiment_matrix.json 一致。

## 0. 环境与证据登记

- [ ] 保存仓库 HEAD、baseline 子模块 commit、WSL/guardenv/Torch/CUDA/GPU 信息。
- [ ] 保存模型缓存、Stage 1/B1 adapter hash、训练配置和评估脚本版本。
- [ ] 保存 NELL23K 与 FB15K237-2 raw/task/eval/rel2text 文件 hash。
- [ ] 为每次运行生成唯一 run ID，并把 seed、bank seed、K、rho、alpha、beta、tau 写入 manifest。

## 1. 本地核心主实验

- [ ] SC_MAIN_CORE_NELL：B1、Random-9、Frequency-matched-Random-9、SC-CL-9。
- [ ] SC_MAIN_CORE_FB：同上，确认本地 QA 与 canonical graph 不混淆。
- [ ] SC_CTRL_ATTRIBUTION：Profile-only、Violation-only。
- [ ] SC_AUX_ERROR：ErrorMemory-9，仅作为辅助对照。
- [ ] 每个方法运行 seeds 2026--2030，保存 EM/OOV/NLL/MRR/Hits 和逐题预测。

## 2. Bank 与泄漏硬门

- [ ] SC_AUDIT_LEAK：完整 train-only bank；valid/test 不得进入候选来源。
- [ ] 检查每条 question ID 一一对齐、正关系唯一、候选互异且数量恰为 K。
- [ ] 检查 train-known collision 为零，并保存候选池大小、采样熵、utility 分位数。
- [ ] 保存 bank JSONL、manifest、audit JSON、脚本版本和 hash。
- [ ] SC_AUDIT_FALSE_NEG：统计持出正例重叠和 OWA 未判定；CoDEx verified negatives 仅作三元组分类审计，不作为关系候选真值。

## 3. 归因、鲁棒性和失败分析

- [ ] SC_CTRL_FREQMATCH：按关系频率分位数匹配随机候选。
- [ ] SC_ABL_UNIFORM / SC_ABL_FILTER / SC_ABL_INFO_NCE。
- [ ] SC_SENS_K：K=3/9/15；SC_SENS_RHO：0/1/3/2/3；SC_SENS_TAU：0.10/0.25/0.50。
- [ ] SC_SLICE_ARITY：一对多、多对一、低度、高度；SC_SLICE_SPARSE：画像稀疏度。
- [ ] 记录 B1 高置信混淆、正例—最难负例 margin、candidate coverage 和生成-排序分歧。

## 4. 数据集扩展

- [ ] WN18RR：完成 SC-CL bank、匹配控制和完整评估。
- [ ] CoDEx-M：官方数据、关系文本和全量 eval QA 已转换；生成/审阅 Stage1/Stage2 task JSON 后，完成 adapter、SC-CL bank、匹配控制和完整评估。
- [ ] canonical FB15K-237、YAGO3-10、CoDEx-S/L 暂缓，主矩阵完成后再评估是否扩展。
- [ ] 每个数据集单独保存版本、hash、转换器、评估脚本和 task count。

## 5. 统计、资源和写回

- [ ] SC_STAT_SEEDS：计算均值、标准差、10,000 次 paired bootstrap 95% CI。
- [ ] 对逐题 reciprocal-rank 差异做配对检验，切片族使用 Benjamini--Hochberg 校正。
- [ ] SC_COST：记录 bank/scoring/Stage 2/inference wall-clock、CPU/GPU peak、tokens、bank rows。
- [ ] 重新生成两张图和所有表，更新 results_status、evidence_ledger、review/review.md。
- [ ] 完成 review：若主效应、泛化或安全门不成立，保留负结果并收窄 claims。

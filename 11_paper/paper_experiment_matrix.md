# SC-CL experiment matrix

2026-10-30 本轮验收调整：按用户要求，先完成 NELL23K、FB15K237-2、CoDEx-M
三个数据集的单 seed=2026 涨点；Stage 1 固定复用，超参数只搜索 Stage 2。
多 seed 统计暂缓。本轮后台队列和验证配置见
`08_experiments/Hard_Negative_Contrastive_GRIP/results/hpo/stage2_three_dataset_20261030/campaign_contract.json`；
排队或验证集提升不计为新的完整测试实测结论。

本矩阵把关键评审问题映射到可执行实验。当前状态不是“全部 preset”：NELL23K 已有
seed=2026 的可审计 SC-CL 实测，FB15K237-2 只有历史 B1/InfoNCE-i0 候选评分，仍没有
当前 SC-CL bank。本文范围限定为知识图谱关系候选排序；CLEGR 的图级问答不纳入当前方法
或实验矩阵。WN18RR 现有一次探索性全测试直接解码输出，但尚缺完整候选重排和公平对照。主数据矩阵为 NELL23K、FB15K237-2、WN18RR 和 CoDEx-M；canonical FB、YAGO3-10、CoDEx-S/L 暂缓。
`local-ready` 表示输入资产已盘点，不表示模型已经运行；`needs-conversion` 表示还需要数据转换、关系文本映射和 adapter/wiring 检查。

| ID | 审稿问题 | 目的 | 数据 | 方法/控制 | 状态 | 主要输出 |
|---|---|---|---|---|---|---|
| SC_MAIN_CORE_NELL | R1/R6 | 核心主效果 | NELL23K | B1 / Random-9 / Freq-Random-9 / SC-CL-9 | observed-single-seed（SC-CL；对照不完整） | EM, OOV, MRR, H@1/3/10, NLL |
| SC_MAIN_CORE_FB | R1/R6 | 核心跨图验证 | FB15K237-2 | 同上 | partial-historical（i0 非 SC-CL） | 同上 |
| SC_MAIN_WN18RR | R2 | 标准小关系集迁移 | WN18RR | B1 / Random-9 / SC-CL-9 | exploratory single-seed generation test observed; ranking/controls pending | EM, MRR, Hits, false-negative audit |
| SC_MAIN_CODEX_M | R2/R3/R7 | 中规模跨图关系候选排序 | CoDEx-M | B1 / Random-9 / SC-CL-9；同一关系候选协议 | data ready; Stage1/Stage2 task inference and model runs pending | EM, MRR, Hits, candidate audit |
| SC_FUTURE_FB_CANON | R2 | canonical FB 控制 | FB15K-237 | 同上；不与 FB15K237-2 合并 | deferred | MRR, Hits, EM |
| SC_FUTURE_YAGO | R2/R7 | 大规模迁移与成本 | YAGO3-10 | B1 / Random-9 / SC-CL-9 | deferred | MRR, Hits, wall-clock |
| SC_FUTURE_CODEX_S | R2/R3 | 可选小规模补充 | CoDEx-S | 单独转换与关系候选协议 | deferred | MRR, candidate audit |
| SC_FUTURE_CODEX_L | R2/R7 | 可选大规模压力测 | CoDEx-L | B1 / Random-9 / SC-CL-9 | optional, deferred | cost, MRR, coverage |
| SC_CTRL_RANDOM | R1 | 均匀负样本对照 | NELL/FB | Random-9 | local-ready | paired delta |
| SC_CTRL_FREQMATCH | R4 | 排除关系频率捷径 | NELL/FB | frequency-matched Random-9 | planned | frequency-bin delta |
| SC_CTRL_PROFILE | R1/R4 | 结构信号归因 | NELL | profile-only | planned | MRR, margin |
| SC_CTRL_VIOLATION | R1 | 模型信号归因 | NELL | violation-only | planned | MRR, margin |
| SC_CTRL_SANS | R1/R4 | 结构负采样参考 | NELL/FB | SANS-style structural control | optional | MRR, H@10 |
| SC_AUX_ERROR | R1 | 错误记忆辅助对照 | NELL/FB | ErrorMemory-9 | implemented-not-run | MRR, EM |
| SC_ABL_UNIFORM | R5 | 均匀下限消融 | NELL | rho=0 | planned | MRR, valid rate |
| SC_ABL_FILTER | R3/R8 | 过滤必要性 | NELL | remove train-known filter | planned-risk | collision, MRR |
| SC_ABL_INFO_NCE | R6 | 候选损失副作用 | NELL/FB | B1 vs no/with InfoNCE | planned | EM, OOV, NLL |
| SC_SENS_K | R5 | 候选数量敏感性 | NELL | K=3/9/15 | planned | MRR, cost |
| SC_SENS_RHO | R5 | mixture floor 敏感性 | NELL | rho=0/1/3/2/3 | planned | MRR, audit |
| SC_SENS_TAU | R5 | utility 温度敏感性 | NELL | tau=0.10/0.25/0.50 | planned | MRR, entropy |
| SC_AUDIT_LEAK | R3/R8 | split/wiring 硬门 | NELL/FB | train-only, known-pair, K=9 | partial-observed（NELL 通过，FB 待运行） | pass/fail, manifest |
| SC_AUDIT_FALSE_NEG | R3 | 开放世界风险 | NELL/FB/WN/CoDEx-M | heldout positive overlap + train-known audit; CoDEx hard negatives separate | planned | train collision, heldout overlap, unresolved OWA |
| SC_SLICE_ARITY | R4/R6 | 一对多/多对一失败切片 | NELL/FB | degree/arity bins | planned | n, MRR, H@10, EM |
| SC_SLICE_SPARSE | R4 | 结构稀疏切片 | NELL/FB | profile sparsity bins | planned | n, margin, MRR |
| SC_STAT_SEEDS | R5 | 随机稳定性 | NELL/FB: 5 seeds; WN/CoDEx-M: >=3, preferably 5 | seeds 2026--2030 where run | planned | mean/std, paired CI |
| SC_COST | R7 | 资源—收益 | NELL/FB/WN/CoDEx-M | bank/scoring/training profiling | partial-observed（NELL） | wall-clock, memory, tokens |
| SC_EXT_BASELINES | R1/R2 | 外部定位 | 标准迁移集 | RotatE/ComplEx/ConvE/SimKGC | optional | filtered metrics, separate table |

固定主协议：候选数 K=9；rho=1/3；alpha=beta=1；tau_s=0.25；
lambda_candidate=1.0；bank 在 Stage 2 前冻结；训练 bank 候选只来自 train vocabulary；
valid/test 不参与训练 bank 采样。主效果只在 matched budget 下解释，外部基线不与生成式
GRIP 指标直接排名。

CoDEx-M 官方 valid/test 硬负例通过替换实体构造，保留作独立三元组分类审计；它们不等价于固定 `(h,t)` 后替换关系的候选负例，也不作为关系候选真值标签。CoDEx-M 主实验使用完整官方正例 split 和 train-only 关系候选过滤。

## 范围决策：CLEGR 不纳入当前论文

CLEGR-Facts/Reasoning 是图级问答任务，答案空间包含不同类型的实体、数值、布尔值和列表；
SC-CL 则在固定实体对上对关系词表采样并优化关系候选排序。二者的反事实单位、候选空间和
主指标不同，故不把 CLEGR 当作 SC-CL 的迁移数据集，也不报告“SC-CL 在 CLEGR 上效果不好”。
用户报告的异机 CLEGR 结果（相对原论文高 `0.415`）仅保留在内部结果状态作为未经核验的
旁支记录，不进入本文论证、实验矩阵或结果表。若未来研究 CLEGR，需另行定义图编辑、答案
验算及 Judge Accuracy 协议，作为独立方法项目处理。

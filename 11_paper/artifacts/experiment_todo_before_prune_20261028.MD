# 结构反事实对比学习：项目待补充实验 TODO

更新时间：2026-10-28（按论文论证职责补充实验；保留原执行清单）  
对应论文：11_paper/main/main.tex  
方法主线：Structure-Counterfactual Contrastive Learning（SC-CL-9）  
实验状态：NELL23K 已完成 seed=2026 的 SC-CL bank、训练、4,943 题候选重排和直接解码，B1 重排有前 200 题；WN18RR 已有 SC-CL bank、训练及验证集选择后的全测试直接解码；CoDEx-M 已完成本地 vLLM train-only task 生成，方法训练产物仍需盘点；FB i0 保持独立历史方法身份。以下新增“论文论证版”作为当前补实验优先级，旧执行清单保留供落地复用。

## 论文论证版补实验方案（2026-10-28，当前执行优先级）

本轮目标是把“实现与一次排序结果”扩展成“候选设计为什么有效、何时有效、需要什么资源”的证据链。优先补齐能支撑论文主线的对照和分析，再安排扩展；不以实验数量或增加篇幅作为验收标准。本次只补计划，没有启动训练、终止现有作业或产生新的方法指标。

### A. 论文思想与实验职责

核心思想：**训练图结构相关性与冻结模型混淆共同决定候选；train-known 过滤明确候选角色；固定候选库让训练信号可追溯；候选级 InfoNCE 学习关系排序边界。**

| 论文问题 | 实验组 | 论证职责 | 优先级 / 论文位置 |
|---|---|---|---|
| 收益来自候选学习还是双信号选样？ | PAPER-EXP-01 | 同初始化、同训练预算的 G0 / Random / Freq-Random / SC-CL | P0；主表 T1 |
| 两个信号各自做什么，组合是否互补？ | PAPER-EXP-02 | Random / Profile-only / Violation-only / SC-CL 因子比较 | P0；主表 T2 |
| 实际选出的候选有哪些不同？ | PAPER-EXP-03 | 结构相似度、违例度、覆盖度与后续排序变化关联 | P0；机制图 F2 |
| 结构画像是否超出关系频率信息？ | PAPER-EXP-04 | 频率匹配、完整画像与频率内打乱画像的对照 | P1；机制图或附录 |
| 过滤处理了哪些已知冲突，哪些事实仍未判定？ | PAPER-EXP-05 | 过滤消融与开放世界三态审计 | P0；审计表 T3 |
| 设计在哪些关系和实体结构上发挥作用？ | PAPER-EXP-06 | 关系频率、train-degree、稀疏度与 B1 混淆切片 | P1；分层效果图 F3 |
| 效果是否依赖某一组十选一干扰项或选项顺序？ | PAPER-EXP-07 | 固定替代候选集、顺序扰动与独立全关系评估 | P1；鲁棒性图 F4 |
| 候选预算和探索下限如何影响训练？ | PAPER-EXP-08 | K、rho、温度的有界敏感性 | P1/P2；附录图 S1 |
| 离线构造、训练与部署评分分别花多少资源？ | PAPER-EXP-09 | 分阶段成本、验证集学习曲线与候选预算前沿 | P1；成本表 T4 / 图 F5 |
| 排序表现与直接解码怎样分别评价？ | PAPER-EXP-10 | 同一模型、两种答案选择规则的完整矩阵 | P0；正文辅表 / 附录 |

PAPER-EXP-00 是执行前的一致性核验，不计作论文实验。上表是待验证问题与预定展示职责，不是预先认定所有实验会呈现优势。

### B. 2026-10-28 实际资产与执行条件

- NELL：`.../results/runs/nell23k_structure_counterfactual_3090_20261021_06_train` 有训练、adapter、直接解码；`.../results/candidate_rerank/nell23k_structure_counterfactual_06_full` 有 4,943 题重排。B1 当前可见重排目录为 `nell23k_baseline_7b_pilot200`，不能把前 200 题当作全测试对照。
- WN：`results/hpo/wn18rr_sccl_seed2026/search_outcome.json` 和 `lambda0p25_final_test/summary.json` 确认验证集选择 lambda=0.25，3,134 题直接解码 EM=86.886%；候选有效率约 99.11%。这与旧辅助 adapter 的 43.17% 是不同运行。当前 runner 的 WN 默认 rank=24、5 epochs，与 NELL rank=4、10 epochs 不同；跨数据集对照应各自在图内匹配。
- CoDEx-M：`04_benchmark/data/codexm/tasks/generation_plan.json` 为 `complete`，本地 vLLM 已生成 236,027 条 context_samples、12,300 条 qa_samples；不是“因为缺 API 仍未生成 task”。模型训练结果需进一步盘点后登记，不把 task 完成当作方法完成。
- NELL K 实验已有资产：`results/hpo/fb_nell_direct_em_20261027/nell23k/k15_parent`、`k3_from_k15`、`k9_from_k15` 均记录 3,253 行且失败数 0。先核对来源与逐题候选前缀，再复用；不要重复为 K 消融运行 B1 全词表评分。
- 另有 NELL 直接解码 HPO：`results/hpo/fb_nell_direct_em_20261027/`，其候选晋级目标是验证集直接解码 EM。该支线与排序论文的验证主指标不同；不能把 HPO 优胜配置悄悄替换原方法，并让其他对照不获得相同调参预算。运行状态以实时日志为准。
- 机器核对：RTX 3090 24 GiB；本轮查询时占用 18,355 MiB、利用率 38%，已有作业使用 GPU。E: 剩余约 36 GiB，C: 约 78 GiB。这是当时快照，不是空闲承诺。正式新作业应串行排队，不能据此文档直接并发启动第二个 7B 作业。
- 正式 GPU Python 固定 `/home/kieran/miniconda3/envs/guardenv/bin/python`；PYTHONPATH 包含 `13_base_method/grip-exp`、HNG `src` 和 `scripts`。0.5B 仅用于入口检查。
- 本阶段可立即开展 CPU 聚合、manifest 核对与 bank 诊断；训练、重排需等可用 GPU。大规模全词表评分、重复 teacher 训练、全参数网格为条件执行项。

### PAPER-EXP-00｜执行前：代码、公式、评分及比较契约对齐

**目的：让所有新增实验测量同一个明确版本的方法。** 以下是本轮读代码发现的差异，先核验实际运行版本，不能直接认定现有论文数值来自当前工作树。

- [ ] PAPER-EXP-00-A：固定主运行的 git commit、采样脚本/模块哈希及原 bank；当前 `counterfactuals.py` 对五维特征全部使用 log1p，结构距离分母为 `1+abs(a)+abs(b)`，头/尾度按关系覆盖的不同实体求均值。论文当前为前三维 log1p、分母 `abs(a)+abs(b)+epsilon`，度描述为边上均值。用真实运行版本决定应修公式还是另立方法版本，不为迎合文字改动旧 bank。
- [ ] PAPER-EXP-00-B：区分三个温度：候选 InfoNCE 温度、采样 softmax 温度、score-violation 平滑温度。当前 fallback violation 在采样函数中使用传入的 temperature；若 score row 有 pairwise_confusion 则直接读取。核对已有 score 字段，记录实际分支和单位，避免改 tau_s 时同时改变两个机制却仍声称单因素消融。
- [ ] PAPER-EXP-00-C：固定 Stage 1 初始化与离线评分 teacher 的独立身份。NELL 现有 run_config 的 s1_adapter / b1_adapter 指向同一路径，不能仅凭目录名判定它是 Stage 1 还是最终 generation-only 模型；核对训练来源、权重哈希及对应 B1 输出。teacher 与最终比较基线使用独立字段命名。
- [ ] PAPER-EXP-00-D：修正评估口径：`evaluate_candidate_rerank.py` 评分题目自带候选；`official_lists.py` 重建选项时包含目标关系。当前是给定候选的十选一排序。保留它作为主协议，同时撤销旧清单中的“已完成 label-blind 候选检索”勾选解释。
- [ ] PAPER-EXP-00-E：增强现有配对聚合。`compare_candidate_rerank_runs.py` 当前只核对 index/target，未核对完整候选列表，且包含 `ours_minus_docs_baseline_em`。新增比较应核对 question_id、完整有序候选列表、prompt/eval hash；禁止将重排 EM 与外部论文直接解码 EM 的差作为训练收益。
- [ ] PAPER-EXP-00-F：从原始 candidate_scores 重算 MRR、H@1、H@3、目标—最难候选 margin。当前 evaluator 的 `score_margin` 是 top-1 减 top-2，不是 gold 减 hardest-negative，字段不能混用；冻结 tie 规则并记录 tie 频数。
- [ ] PAPER-EXP-00-G：FB 精确区分原关系 ID 与关系文本，核对 alias 合并（代码提示 237 ID 对应 233 个文本标签）对候选去重、train-known 过滤与指标的影响。实际数目以固定版本统计为准。

**验收**：形成 `protocol_contract.json`，包括模型与初始化、teacher、task/eval、特征/距离、温度语义、优化步与微批顺序、采样 seed、候选/答案规则和指标定义。同一比较表引用相同 contract；不同训练方法记录各自 bank hash。

### PAPER-EXP-01｜同契约主效果：把候选损失与候选选样分开

**问题**：相同初始化和训练预算下，结构—模型选样相对普通候选学习提供什么额外表现？

- [ ] PAPER-EXP-01-A：先补现有 B1 的全量 4,943 题重排，作为低成本配对诊断。它与历史 SC-CL 的比较仍保留预算条件，不自动成为严格归因对照。
- [ ] PAPER-EXP-01-B：NELL 四臂：G0-matched（lambda=0）、Random-9、Frequency-matched-Random-9、SC-CL-9。G0 与候选训练臂共享 Stage 1、完整 task、优化器、训练步、微批顺序与 LoRA；候选臂共享 K=9、lambda=1、候选温度 1、train-known 过滤。G0 的前向成本较少，应记录实际 FLOPs/时间，不能称所有臂 GPU 时长相同。
- [ ] PAPER-EXP-01-C：Random 使用同一合法池均匀无放回采样；频率匹配随机在每个问题中匹配 SC-CL 九个候选的 train-frequency 桶配额。预先冻结训练图频率桶，记录短缺与回退；不能挑随机 seed 或放宽过滤补配额。
- [ ] PAPER-EXP-01-D：不同方法各自固定候选库，同一方法跨训练 seed 复用自身 bank；共用的是问题、候选池与评估集合。若用 `rho=1` 或 `alpha=beta=0` 复用 SC sampler 实现 Random，先验证确为均匀采样、provenance 命名与独立参考 sampler 等价；不能把随机库仍标成双信号 SC-CL。
- [ ] PAPER-EXP-01-E：种子先用 2026 完成执行与比较闭环，再对主四臂补 2027/2028；五种子 2026–2030 为后续增强。固定样本数/种子计划，不根据显著性提前停种子或选 seed。
- [ ] PAPER-EXP-01-F：跨图首先复用 WN 完整训练资产并补其本地 B1 核验；同时盘点已完成 task 的 CoDEx-M，作为第二个关系空间验证。FB 当前协议缺完整 adapter 时，先做资产恢复/同契约 G0，不用 i0 代替 SC-CL。根据实际就绪程度选择第二数据集，选择规则在查看新增测试差异前固定。

**固定项**：测试集全量、相同候选选项和顺序、prompt、截断长度、score 归一化、直接解码 max_new_tokens；模型选择只依据固定验证指标，若调参所有主臂获得相同搜索预算。旧 WN/HPO 的不同解码长度单独记录。

**输出**：T1 各数据集/方法/seed 的 H@1、MRR、H@3 与 paired delta；T1b 直接解码矩阵；`paired_question_metrics.jsonl`、`alignment_report.json`。期望检验 SC-CL 的排序优势，观察结论由实测给出。

**工作量**：B1 只需新重排；G0/随机/频率随机需新训练。SC-CL 旧 seed=2026 仅在 EXP-00 契约吻合时复用。对照与数据集映射到 `SC_MAIN_CORE_NELL`、`SC_MAIN_CORE_FB`、`SC_MAIN_WN18RR`、`SC_MAIN_CODEX_M`。

### PAPER-EXP-02｜双信号互补：结构画像和模型混淆分别贡献什么

- [ ] PAPER-EXP-02-A：同一采样实现、rho=1/3、K=9 下运行 Random、Profile-only（alpha=1,beta=0）、Violation-only（alpha=0,beta=1）、SC-CL（alpha=beta=1）。Random 和 SC-CL 复用 EXP-01，新增两个单信号臂；所有臂使用相同 Stage 2 契约。
- [ ] PAPER-EXP-02-B：报告 SC-CL 对两个单信号臂的 paired delta；同时统计四模型逐题 RR 与正确性。比较“Profile 改善、Violation 未改善”等类别，展示组合方法是否覆盖不同类型的问题，不只展示一行平均值。
- [ ] PAPER-EXP-02-C：若报告交互量，预先定义 `I = MRR_SC - MRR_Profile - MRR_Violation + MRR_Random`，按训练 seed 和相同问题计算区间。只有对应证据支持时称“超加性”；仅优于两单臂不能自动推出统计交互。
- [ ] PAPER-EXP-02-D：检查各臂选样难度与 utility 尺度。双信号相加可能使 softmax 更集中；先报告熵和 violation 分布，若差异很大，再加验证集确定的熵匹配/难度匹配单信号控制，区分信号信息和分布锐度。

**输出**：T2 单信号与组合对照；F1 逐题覆盖/互补图，附 bank 熵和 utility 尺度。当前参数可用 `COUNTERFACTUAL_STRUCTURE_WEIGHT` / `COUNTERFACTUAL_VIOLATION_WEIGHT` 复用 builder；新增臂仍需独立 metadata 与训练。

**统计要求**：单 seed 为机制描述；要将“互补”写成稳定结论，应至少扩展两个新增单信号臂至同样的三个训练 seed。映射 `SC_CTRL_PROFILE`、`SC_CTRL_VIOLATION`。

### PAPER-EXP-03｜候选分布与学习边界：解释方法到底选了什么

这是最高性价比的机制分析，优先使用已有冻结 score、profile、bank 与逐题输出，通常不需重新训练。

- [ ] PAPER-EXP-03-A：对完整合法候选池与 Random、Freq-Random、Profile、Violation、SC-CL 的已选候选分别统计：结构相似度、冻结 B1 violation、score gap、train-frequency、关系覆盖、每题池大小；按问题等权汇总，避免大池样本支配统计。
- [ ] PAPER-EXP-03-B：画结构相似度×violation 的联合分布或等额抽样散点；报告高结构/高混淆、低结构/高混淆等象限中的选择比例。阈值来自训练池固定分位数，不能看测试收益再划线。
- [ ] PAPER-EXP-03-C：区分“候选分布熵”和“抽中九个候选后的覆盖度”。计算 `H(p)=-sum p log p`、有效候选数 exp(H)、全训练候选关系覆盖、跨问题重复率。熵描述构造分布，九个候选互异本身不证明多样性优于其他方法。
- [ ] PAPER-EXP-03-D：在独立验证问题的相同评估候选上计算 gold-minus-hardest margin、RR 及其相对 G0/Random 的变化；已有冻结 teacher 的低 margin 问题用于描述性切片。训练 bank 上的边界变化单独标为训练诊断。
- [ ] PAPER-EXP-03-E：对照同一个训练问题中的“入选/未入选”候选分数及规则记录，抽取可完整复算的实例。若要将训练候选分布关联到测试表现，按目标关系分组做描述性关联，不声称训练样本与测试题存在未经核验的一一因果对应。

**输出**：F2 双信号候选分布；附表覆盖、熵、gap；2–3 个逐项可复算的构造实例。映射 `SC_CTRL_PROFILE`、`SC_CTRL_VIOLATION`、`SC_AUDIT_LEAK`，新增机制分析 ID 保留为 PAPER-EXP-03。

### PAPER-EXP-04｜结构信息特异性：排除‘只是选高频关系’

- [ ] PAPER-EXP-04-A：先完成 EXP-01 的频率匹配随机；报告各训练方法频率桶分布及匹配误差，建立比较是否真的匹配的证据。
- [ ] PAPER-EXP-04-B：在 train-frequency 桶内随机置换关系画像，保留真实 B1 分数、目标问题、已知关系过滤和相同采样参数；冻结置换 seed。比较完整画像+violation 与置换画像+violation，检查真实结构对应关系的作用。置换可削弱而非完全消除结构，解释按实际设计限定。
- [ ] PAPER-EXP-04-C（条件执行）：若频率匹配后需要进一步解释，再加 Frequency-only+Violation（仅保留 log edge_count）与完整五维画像+Violation。只改特征项，明确特征维数改变如何影响距离归一化。

**输出**：结构信息对照表或 F2 附面板，承担结构画像定位的论证职责。频率匹配为 P0，打乱画像为 P1，新增特征训练为条件执行；不预设“结构特征一定优于频率”。映射 `SC_CTRL_FREQMATCH` / `SC_SLICE_SPARSE`。

### PAPER-EXP-05｜开放世界候选审计与 train-known 过滤职责

- [ ] PAPER-EXP-05-A：对各方法的训练候选同时审计三态：train-known 冲突、同实体对的 valid/test 已知正例重叠、其余开放世界未判定；按候选数与问题数分别报告分母。持出正例重叠是已知正例冲突的可观察部分，不是所有潜在假阴性的真实比例。
- [ ] PAPER-EXP-05-B：仅在 CPU 中移除 train-known 过滤，保留排除本题 gold 与相同采样规则，比较有/无过滤的候选冲突和覆盖。没有新增可选的其他已知关系时，该诊断可能均为 0；不要据此捏造过滤的经验收益。
- [ ] PAPER-EXP-05-C：构造有证据的冲突实例卡：实体对、目标关系、候选、训练/持出事实出处、过滤判定、结构分数与违例度。按固定 seed 从各已存在类别抽样，明确总体规模，不能只选能讲故事的案例。
- [ ] PAPER-EXP-05-D（可选压力诊断）：仅从 train 中预先隐藏一部分已知正例，重算采样图的画像与过滤，量化这些已知但被隐藏的事实被选为负候选的比例。固定 teacher 若曾见过这些事实，应明确此实验是采样压力诊断；若要验证未见事实风险，需重新构造 task/teacher，成本另计。

**禁止回流**：持出审计只描述已冻结 v1 的状态。它不参与训练候选过滤、采样参数、阈值或模型选择；若据此设计 v2，应独立版本、训练和确认评估。CoDEx 的官方实体替换 hard negatives 不给关系替换候选提供真值标签。

**输出**：T3 各方法已知冲突与未判定计数；过滤前后对照；可核验实例。CPU 审计优先，无需先训练关闭过滤的高风险模型。映射 `SC_AUDIT_FALSE_NEG`、`SC_ABL_FILTER`、`SC_AUDIT_LEAK`。

### PAPER-EXP-06｜结构切片：把平均收益落实到方法定位

- [ ] PAPER-EXP-06-A：训练图定义关系频率分位、头/尾实体度、关系 arity（1–1/1–N/N–1/N–N）、画像稀疏度；固定分桶与阈值，test 仅用于对应分桶计分。arity 的定义与采用阈值写入配置。
- [ ] PAPER-EXP-06-B：按 G0/B1 对同一候选集合的 margin 或正例 rank 定义混淆难度层。这是模型条件的描述性切片，与 train-only 图切片分别展示，不用它筛掉总体错误题。
- [ ] PAPER-EXP-06-C：报告每层 n、关系数、G0/Random/Freq-Random/SC-CL 的 H@1、MRR、paired delta 和区间，并补按关系等权的 macro-MRR，防止高频关系掩盖覆盖范围。
- [ ] PAPER-EXP-06-D：输出完整分层结果和各层 question ID。小样本层采用描述性区间/计数；合桶规则在读取测试差异前固定，不只展示 SC-CL 占优的层。

**输出**：F3 按结构切片的配对效应图；附录全表。检验的预期是“结构相关与模型混淆存在时有可辨别的作用模式”，不预承诺所有层都领先。通常仅需 CPU 聚合已有全量分数。映射 `SC_SLICE_ARITY`、`SC_SLICE_SPARSE`。

### PAPER-EXP-07｜候选集合与顺序鲁棒性

- [ ] PAPER-EXP-07-A：对固定问题和同一十候选集合，采用 3 个预先冻结的排列 seed，重新构造 prompt 并重新评分至少 Random 与 SC-CL。已有分数排序后换行顺序不能替代该实验，因为候选列表本身是模型输入的一部分。
- [ ] PAPER-EXP-07-B：保留标准十选一为主结果，增加预先构造的 Random / Frequency-matched / train-profile-similar 干扰项集。各模型在同一组候选和顺序上评估；这是目标关系加干扰项的条件排序，不声称 label-blind 检索。
- [ ] PAPER-EXP-07-C：资源允许时在固定哈希抽样的验证集、随后预登记的测试子集上做全 train-relation vocabulary 排序。候选词表独立于 gold，gold 仅用于最终 rank；prompt 不应继续暴露只含 gold 与九干扰项的旧列表。单列为新评估协议，不与原十选一数值混排。
- [ ] PAPER-EXP-07-D：若采用 label-blind top-L 检索再重排，同时报告 gold recall@L、条件 MRR 和将漏检计作失败的端到端指标；不能只评估 gold 已被召回的子集。关系画像依赖 gold 的构造规则不能直接用于该检索器。

**预算约束**：先对固定 500–1,000 题和两个主候选模型做顺序/干扰项检查，子集选择独立于结果。NELL 198 关系的全词表评分规模约为十选一的 19.8 倍 continuation 数，prompt 长度/截断也可能变化，先测吞吐再排程。WN 仅 11 个原始关系，全关系与十选一差别较小，不用它替代大词表鲁棒性。

**输出**：F4 候选条件/顺序下的 MRR 与 H@1；独立全词表表 S2；候选数、覆盖与截断统计。属于新增辅助协议 PAPER-EXP-07，不覆盖现有主协议。

### PAPER-EXP-08｜有界敏感性：候选预算与均匀探索

- [ ] PAPER-EXP-08-A：NELL K=3/9/15，共享 K15 parent 的固定候选前缀；先比较派生 K9 与历史 K9 的逐题候选及 score-source hash。相同候选才可复用原模型，metadata 不同不等于候选不同。当前三个库已存在，新工作是核验和模型训练。
- [ ] PAPER-EXP-08-B：rho=0、1/3、2/3，保持信号权重及其他温度语义固定；同时报告熵、覆盖、已知冲突状态、MRR。rho=1 的 Random 已由 EXP-01 承担，不重复算作新的机制证据。
- [ ] PAPER-EXP-08-C（条件执行）：采样温度 0.10/0.25/0.50 或候选损失权重 0.25/0.5/1.0，先在验证集做单轴实验；EX00-B 确认温度无耦合后才能称单轴。不要做 K×rho×tau×lambda 全笛卡尔积。
- [ ] PAPER-EXP-08-D：WN 有效池不足 15 时不运行 K15，按过滤后最小池大小选择可行 K；各数据集独立注明 train vocabulary 与有效池范围。训练 K 与测试候选数是两个不同变量。

**输出**：S1 敏感性图与 K–成本图。固定 task/QA 轮数的 K 实验因 continuation 数不同，应报告真实 token/forward/耗时；若要声称预算效率，另加等候选 forward 预算比较。映射 `SC_SENS_K/RHO/TAU`、`SC_ABL_UNIFORM`。

### PAPER-EXP-09｜分阶段成本与验证集学习曲线

- [ ] PAPER-EXP-09-A：分别计量图统计 CPU、teacher 全关系评分、freeze/wire、Stage 2、直接解码、重排，报告 GPU 时间/峰值、CPU 内存、tokens、candidate_forwards 和产物体积。已记录的 NELL 评分 19,027 秒、训练约 80,870 秒、重排 3,567 秒作为该次测量锚点。
- [ ] PAPER-EXP-09-B：同方法复用 teacher 分数和候选库时，分别报告单次预处理成本与摊销后的多次训练成本；不要把多 seed 共享的预处理成本计入每个训练 arm，或在总成本里完全漏掉它。
- [ ] PAPER-EXP-09-C：新增训练在固定 25%/50%/100% 优化步上保存验证集 H@1/MRR 与直接解码 EM，报告相同优化步和实际 wall-clock 两种横轴。中间 checkpoint 只需保留可追溯指标和必要权重，不假设旧 checkpoint 已存在；完整 test 只用于冻结模型的最终报告。
- [ ] PAPER-EXP-09-D：用真实成本与实际 paired delta 画前沿。若没有时间优势，就陈述资源条件与达到的排序表现，不承诺“更快收敛”或“低成本”。当前磁盘紧张，不复制基座权重，不自动清理其他实验 checkpoint。

**输出**：T4 三阶段/多阶段成本表，F5 验证集表现–训练步/时间曲线，K–成本附图。映射 `SC_COST`。新训练过程采样即可补曲线，通常不需为曲线另训全部模型。

### PAPER-EXP-10｜训练目标×答案选择：完整报告排序与文本输出

- [ ] PAPER-EXP-10-A：对 G0、Random、Freq-Random、SC-CL 的同一冻结 checkpoint，同题分别直接解码与逐候选评分，生成方法×答案规则矩阵。EXP-01 结果直接复用，不另开训练。
- [ ] PAPER-EXP-10-B：分别统计 exact match、答案在本题候选列表内的比例、在 train relation vocabulary 内的比例，以及“生成错/重排对”“生成对/重排错”。列表外答案与词表外答案分别命名；已有 candidate_invalid 不是天然等于 OOV。
- [ ] PAPER-EXP-10-C：量化 model 差异和 decoding 差异时保持同一规则。SC-CL 自身 91.52% 重排 H@1 与 87.03% 直接解码 EM 的差值属于答案选择变化；只有与同规则对照的差异才能讨论训练表现。
- [ ] PAPER-EXP-10-D：直接解码作为完整输出行为报告，不因为排名不占优而删除；也不把“不显著”解释成等效或无损。若要提出非劣效结论，另行预设有实用依据的界值与对应统计设计。

**输出**：方法×选择规则矩阵、逐题迁移计数、必要的输出实例。映射 `SC_ABL_INFO_NCE`。它的职责是界定候选学习与答案选择，正文主叙事仍围绕候选排序。

### C. 统计与模型选择的统一规则

- [ ] PAPER-STAT-01：主指标预先选 MRR，H@1/H@3 为辅助；同时给 micro 和按关系等权 macro 指标，说明量纲与百分点。不同测试候选数单列。
- [ ] PAPER-STAT-02：同一方法的 bank seed 固定，训练 seed 独立；模型比较使用同 seed、同问题配对。训练随机性以每 seed 的结果和 mean±sample-std 报告，不能把同 checkpoint 的测试 bootstrap 当作多种子稳定性。
- [ ] PAPER-STAT-03：已有单 seed 可做 10,000 次问题级 paired bootstrap，仅反映给定模型的题目抽样变化；存在重复实体对/强关系簇时补实体对聚类重采样。多 seed 用同时保留模型配对和问题配对的分层/双轴 bootstrap，交代 seed 数少时的局限。
- [ ] PAPER-STAT-04：核心比较先定 SC-CL–Random、SC-CL–Freq-Random；对两项主检验预先选择 Holm 校正，探索性多切片采用 BH 或明确只报告区间。不在结果出来后变换检验与主指标。
- [ ] PAPER-STAT-05：切片阈值、最小实际差异、候选构造与调参预算在读取新增测试结果前固定。路线决策优先基于 validation；既有测试已被观察，不再宣称它是未揭盲确认集。若基于其观察改设计，明确为后续探索，并用新的固定协议/独立数据验证。

### D. 代码复用与需要补的实现

| 组件 | 已有入口 | 本次计划中需要补什么 |
|---|---|---|
| 训练与记录 | `configs/run_error_memory_wsl.sh`、`scripts/train_listed_contrastive.py` | shell 目前仍只接受 error_memory/structure_counterfactual；增加语义明确的 Random/Freq-Random 控制、同契约 G0 与全部 provenance |
| bank 构造 | `scripts/build_structure_counterfactual_bank.py`、`src/hard_negative_grip/counterfactuals.py` | 复用已有分数；固定温度语义；新增频率匹配与打乱画像规则；不同方法独立标识 |
| K 派生 | `scripts/derive_structure_counterfactual_bank.py` | 已支持 CPU 前缀派生；补候选前缀等价与训练结果对照 |
| 模型评分 | `scripts/evaluate_candidate_rerank.py` | 完整候选/顺序/prompt 哈希对齐、指标独立重算；新候选协议需显式重建输入 |
| 配对比较 | `scripts/compare_candidate_rerank_runs.py` | 移除跨论文不同答案规则的增益计算；增加候选一致性、RR、gold-margin、bootstrap 和多 seed 汇总 |
| 审计与分层 | `scripts/verify_error_memory_wiring.py`、已有 bank/score | 新增只读审计、频率/degree/arity 切片、选样分布统计，输出逐条证据 |
| 第二数据集 | WN / CoDEx-M pipeline | 盘点已有模型产物、核对 WN 基线；CoDEx 已完成 task，无需重新走 API 生成 |

HNG 均指 `08_experiments/Hard_Negative_Contrastive_GRIP/`。本表列“需要补”不代表代码已经实现；本轮不修改训练脚本。

### E. 按实际资源安排，避免一次启动全部网格

历史 NELL 单个 SC-CL 完整训练约 80,870 秒，即 22.46 GPU 小时。以下是按这一测量锚点估算的**等效训练量**，不是新实验实测耗时；G0 实际较轻，其他数据集与不同配置须独立测量。均不含排队、teacher、评估与工程时间。

| 批次 | 内容 | NELL 新训练等效量 | 先得到什么 |
|---|---|---|---|
| 立即 CPU 批次 | EX00、EX03/05/06 的可复用部分、指标聚合与随机库设计 | 0 | 一致性、候选分布、已知冲突与现有模型的描述性切片 |
| 第一批 GPU | B1 full rerank；G0/Random/Freq-Random/Profile/Violation 各 seed=2026，复用兼容的 SC-CL | 5 个新训练，约 112.3 小时 | 四臂主比较与双信号机制；未完成模型的切片等其输出 |
| 第二批确认 | 主四臂扩展至 2026/2027/2028；两个单信号臂暂为单 seed | 总共 14 个 arm×seed，若复用旧 SC-CL 则新增 13 个，约 292 小时 | 主效果三种子与描述性机制；不能把单信号单 seed 称稳定互补 |
| 机制增强 | 单信号 Profile/Violation 也补至三个 seed | 再新增 4 个，约 89.9 小时 | 六臂三种子；累计新增 17 个，约 381.9 小时 |
| 第二数据集 | WN 补全评估/同预算控制；或已就绪的 CoDEx-M 当前方法与随机控制 | 单独测量 | 跨关系空间验证，模型数量与 seed 如实标记 |
| 条件扩展 | 画像打乱、K/rho、全词表评估、五种子 | 按剩余资源单独排程 | 针对仍未解释的问题扩展，不做全网格 |

上述复用以 EX00 一致性核验为前提，若旧 SC-CL 不能合并，新增训练量需再加一次。第二批“总共 14”是四主臂×3 seed 加两单信号臂×1 seed，包含第一批，并非第一批之外再训练 14 次。24 GiB 单卡按串行计划安排；不自动停止正在运行的 HPO/CoDEx 作业。

最先执行的顺序：EX00 → B1 全量重排和 CPU bank/OWA 分析 → 统一 Random/G0 入口 → seed=2026 主四臂及两单信号 → 主四臂三种子与第二数据集 → EX06/07/09/10 图表 → 按机制问题选择 EX04/08。旧路线门只作为资源决策，正式扩展不依赖单 seed 达到显著性。

### F. 论文写回与逐组完成定义

每组结果都保存：`exp_id`、问题、变化因素、固定条件、baseline/model ID、contract hash、bank/teacher hash、训练 seed、样本 ID、原始输出、指标与资源、比较是否匹配、实际结论、`section_id`、`item_id`、`claim_links`、`paper_role`、主文/附录位置。EX01 对应方法效用，EX02–04 对应双信号机制，EX05 对应来源/真值审计，EX06–10 对应适用模式、鲁棒性与资源条件。

- [ ] 每组完成后更新 `11_paper/paper_experiment_matrix.md/json`、`results_status.md`、`artifacts/evidence_ledger.md`；新 ID 先登记映射，不能只在本文件勾选。
- [ ] T1/T2/T3/T4 与 F1–F5 是预定显示目标；数值与图形只从真实产物生成，不使用占位数字。
- [ ] 正文采用“问题→设计→最硬的结果→机制解释”的顺序。每个表图写清论证职责；必要条件放定义、表注或集中讨论。未支持的优势不强写，负向/无差异结果保留在对应完整表和内部决策记录。
- [ ] 首轮足够支持写回的最低包：同契约全量 NELL 主对照、两个单信号、候选分布、OWA 审计、完整直接解码与成本。跨种子和第二数据集完成到什么层级就报告到什么层级，不用“主表变长”替代主张验证。

### G. 外部方法定位依据（辅助实验设计，非本文新结果）

- [SANS，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.492/) 使用节点 k-hop 邻域构造结构感知负样本。因此可借鉴其结构采样问题，但本文的关系画像对照应按关系替换重新定义，不能把实体候选指标直接拿来排名。
- [ProGCL，ICML 2022](https://proceedings.mlr.press/v162/xia22b.html) 在 GNN 图对比学习中分析困难样本与假阴性问题。这支持分别检查难度与真值状态的实验动机，不证明 SC-CL 已降低假阴性风险，也不是直接可复用的关系排序基线。
- [CoDEx 官方项目](https://github.com/tsafavi/codex) 区分 link prediction 与 triple classification，并提供人工核验 hard-negative triples。本文关系替换候选的真假仍须按自己的候选空间审计。

下方保留原工程执行清单；涉及优先级、状态、模型选择或评价口径时，以本节的当前方案与实际产物为准。原始文档已备份到 `11_paper/artifacts/experiment_todo_before_20261028.MD`。

---

## 历史路线决策（2026-10-23，当前优先级以上方论文论证版为准）

- 暂沿用现有 SC-CL-9 作为 v1，不立即改算法；论文定位收窄为“开放世界知识图谱关系候选排序的可审计结构反事实对比学习”。
- [x] SCOPE-CLEGR-001：CLEGR-Facts/Reasoning 属于图级组合问答，答案空间、反事实对象和 Judge Accuracy 均不同于 SC-CL；从当前论文与实验矩阵排除。
- [x] 不把未经核验的 CLEGR 异机 +0.415 报告写成论文结果，也不把不适配的 SC-CL 结果表述为“CLEGR 上效果不好”。
- 先在 NELL23K 补齐同预算 B1、Random-9、frequency-matched Random-9，以及 Profile-only/Violation-only 归因对照，并完成 held-out positive overlap 与候选风险审计。
- 审计主要复用现有 bank、逐题候选分数和 test 预测；新对照臂仍需按统一训练契约训练。现有 NELL SC-CL seed=2026 优先复用；复用前须确认初始化、训练配置、候选库与实际执行版本吻合，契约不一致时作为独立历史结果保留。
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
| WN18RR | 除低预算辅助运行外，已有 2,709 行 SC-CL bank 与 lambda=0.25 获选模型，3,134 题直接解码 EM=86.89% | 补完整重排、同预算对照并核对本地 B1 复现差异 |
| CoDEx-M / YAGO3-10 / CoDEx-S/L | CoDEx-M 官方数据与本地 vLLM task 生成已完成；其他扩展暂缓 | 优先盘点 CoDEx-M Stage 1/2 产物后补对照，不再以缺 API 作为当前阻塞原因 |
| CLEGR | 本地低预算 GRIP task/adapter 与 test 评估已完成；用户另报告异机运行较原论文高 +0.415 | 数据集和两类 CLEGR 结果均排除于当前论文；内部保留来源记录，不将其解释为 SC-CL 效果 |
| 0.5B smoke | 只用于验证环境/入口/显存和导入路径 | 不能作为论文主结果、不能与 Qwen2.5-7B 正式结果混合 |

论文中的 NELL 单种子指标来自可复核原始文件；FB i0 是用户确认的有效主运行，候选评分文件
仍可复核，但原始运行记录缺失；旧 WN18RR/CLEGR 低预算生成 EM 仅是任务适配器辅助指标；WN 新增的 SC-CL 获选模型结果与旧辅助运行分别记录。CLEGR 异机运行的 +0.415 仅作内部来源记录，因任务不在本文范围且原始证据未同步，
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

- [x] 已有评估分为训练 bank diagnostic 与固定十选一候选重排，训练诊断不替代测试指标。
- [x] 2026-10-28 口径更正：NELL evaluator 读取题目自带的十个选项；official_lists.py 重建选项时以 gold 加九个候选再打乱。这是给定候选的条件排序基准，不是 label-blind 检索，仍可作为本文关系候选排序主协议。
- [ ] 不依赖 gold 构造候选的全关系排序或检索另立辅助协议（PAPER-EXP-07）；与原十选一结果分别报告。
- [~] 已有逐题 predictions.jsonl 与 candidate_scores / gold_rank；以下为统一目标 schema，缺失字段需从原始分数补算并核验：
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
- [~] 已有 summary.json 的 EM、count 和时间等记录；以下为统一汇总目标，MRR、gold-margin、tie 等字段需核验或补全：
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
- [x] 2026-10-28 已确认 runner 支持 LISTED_NEGATIVE_K，且存在 derive_structure_counterfactual_bank.py；NELL K=15 parent 与 K=3/9 派生库均有 3,253 行、rows_failed=0 的元数据。
- [ ] 核对实际 requested_k/actual_k、wiring 和训练预算；已有候选库不等于已完成 K 敏感性模型实验。

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
- [ ] CoDEx 官方实体替换 verified negatives 仅作独立三元组分类参考；关系替换的冲突审计必须查询相同实体对与关系标签，不能用实体替换负例估计关系候选 purity。

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
- [ ] bank seed 固定为 2026；同一方法跨训练 seed 复用其自身 frozen bank。不同采样方法各用独立 bank，共享训练问题、可用候选池、随机种子规则及测试候选；不能让各方法使用同一份已选负候选。
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
- [x] 当前 runner 已有 WN18RR 分支，SC-CL bank、训练及 lambda=0.25 的全测试直接解码结果已存在。
- [ ] 为获选模型补齐完整重排与匹配控制，并核对其与现有 B1 的初始化、解码长度和预算。
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
- [ ] 实体替换 verified negatives 如使用，独立报告其三元组分类指标；关系候选分别报告已知正例冲突、开放世界未判定与 MRR/Hits，不混合两种空间计算 purity。
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
9. WN18RR 补获选模型重排和匹配控制；CoDEx-M 已完成 task 生成，优先盘点并复用 Stage 1/2 后补同契约结果。canonical FB、YAGO3-10、CoDEx-S/L 暂缓；CLEGR 不纳入本文；
10. 根据真实且可追溯的结果更新统计汇总、证据账本和论文；不将预设数字、单种子筛查或旧方法结果写成当前方法结果。

---

## 14. 明确不做

- [ ] 不把 0.5B smoke 当成论文结果。
- [ ] 不用 smoke task、64-QA 子集或 underfit bypass 生成主结果。
- [ ] 不把 valid/test 三元组用于训练 bank 的 profile、frequency、teacher 评分问题来源或 known-pair filter；测试评分用于正式评估，持出事实仅用于已冻结候选的事后审计。
- [ ] 不把 train 中未观测关系称为真实负例。
- [ ] 不把 FB15K237-2 与 canonical FB15K-237 重复计权。
- [ ] 不把 CLEGR 的用户报告 +0.415 或本地低预算 EM 移入当前论文，也不以不匹配的关系候选方法得出 CLEGR 负结果。FB i0 保留 InfoNCE-i0 方法标签和 provenance 缺口。
- [ ] 不在没有原始证据时填入显著性、置信区间、资源耗时或跨数据集结论。

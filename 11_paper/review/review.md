# 模拟 ICLR 审稿报告：结构反事实对比学习

日期：2026-10-23  
审阅类型：独立 reviewer-first pass；结论基于当前稿和可核查项目文件，不把预设数字当作证据。

## 总评

论文的中心问题清楚：在 GRIP 的候选关系学习阶段，随机候选可能过于容易，单纯依赖
模型分数的困难候选可能引入已知关系冲突或开放世界假阴性。SC-CL 的方法边界也比较
干净：训练图关系画像、冻结 B1 violation、known-pair 过滤、固定 K 的离线 bank 和
候选 InfoNCE。当前版本已经纳入 NELL23K seed=2026 的正式运行与历史 FB 候选评分，
但仍缺少同协议完整基线、五种子、FB 当前 SC-CL 和跨数据集转换证据，不能作为已完成
投稿结果。

模拟评分：方法新颖性 6/10，实验充分性 4/10（补齐协议后可达 7/10），清晰度 7/10，
可复现性 4/10，当前总体建议为 **大修后再审**。

## Reviewer 1：归因和公平性

**问题。** SC-CL 是否只是增加了一个候选损失，或只是利用了关系频率/更难负例？
如果 Random-9 的候选池、训练预算、初始化、adapter 或采样种子不同，主结论无法归因。

**论文中的回应。** 主矩阵加入 B1、Random-9、Frequency-matched Random-9、Profile-only、
Violation-only、ErrorMemory-9 和完整 SC-CL-9；所有直接对照共享 base model、Stage 1
adapter、prompt、K、训练预算和五个种子。只把 SC-CL 相对频率匹配随机的配对差异解释为
候选构造效应。

**需要的证据。** 每个 seed 的 config、bank hash、question-ID 对齐表、候选频率分位数
匹配报告和原始预测。

**会推翻主张的结果。** 单一信号或频率匹配随机与完整方法相同；此时贡献必须收窄为
基线协议或单信号观察。

## Reviewer 2：数据集广度和泛化

**问题。** NELL23K 和本地 FB15K237-2 是否足以支持跨图泛化？标准 FB15K-237、
WN18RR、YAGO3-10 与 CoDEx 的实体/关系文本和任务格式是否一致？

**论文中的回应。** 数据集分为本地核心层、标准迁移层和 CoDEx 假阴性压力层。
正文明确 FB15K237-2 与 canonical FB15K-237 不重复计权，待转换数据只有在 task JSON、
关系文本映射和 adapter 通过检查后才能进入主表。

**需要的证据。** 每个数据集的版本和 hash、三元组到 QA 的转换器、关系文本覆盖率、
adapter hash、完整 test QA 数量和 split overlap 检查。

**会推翻主张的结果。** 方法只在本地格式有效；结论应改为“当前 GRIP 任务格式上的
候选学习设计”，不能写成通用跨数据集方法。

## Reviewer 3：开放世界假阴性与数据泄漏

**问题。** train-known 过滤只说明候选未在训练图出现，并不能说明它在真实世界中为假；
结构相近的关系可能在 valid/test 或外部知识源中同时成立。

**论文中的回应。** 方法定义把“反事实”限定为训练图上的受约束标签替换，不使用
“真实负例”措辞。每个 bank 运行 train-known collision、持出正例重叠和 CoDEx
verified-negative 审计；OWA 未判定单列，不计作错误率。

**需要的证据。** bank 中每个候选的来源、外部/CoDEx 对齐结果、collision 分母、开放
世界未判定比例以及 valid/test 入 bank 的硬门日志。

**会推翻主张的结果。** SC-CL 的持出正例重叠显著高于随机且没有排序收益；此时需要
软权重或降低方法主张，不能通过删除冲突样本掩盖风险。

## Reviewer 4：统计稳定性和结果真实性

**问题。** 预设的均值、标准差和增益看起来像结果，是否会诱导选择性报告？
单一 seed 或无 paired uncertainty 不能支持“稳定提升”。

**论文中的回应。** 正文把 NELL 单种子实测与历史 FB i0 分开标注；正式协议固定五个
训练种子 2026--2030，报告均值、标准差、10,000 次 paired bootstrap 95\% 区间，
对切片检验做多重比较校正。预设表不写 p 值或显著性星号。

**需要的证据。** 原始预测、每题 reciprocal rank、seed-level config、bootstrap 脚本
和完整失败样本清单。

**会推翻主张的结果。** 置信区间跨越实际不利效应，或不同 seed 的效应方向不稳定；
结果应报告为不确定/无明显收益。

## Reviewer 5：基线、效率和生成副作用

**问题。** 候选排序改善是否牺牲自由生成？离线 B1 全关系 scoring 和 bank 构造成本
是否值得？外部 KGC 基线能否公平比较？

**论文中的回应。** 分开报告 EM/OOV/NLL 与 MRR/Hits；资源表记录 bank、scoring、
Stage 2 和推理成本。RotatE、ComplEx、ConvE、SimKGC 只在统一转换与 filtered protocol
通过后作为外部参考，不与 GRIP 生成式指标混合排名。

**需要的证据。** tokens、wall-clock、CPU/GPU 峰值、bank 行数、checkpoint hash、
同一硬件和精确外部基线配置。

**会推翻主张的结果。** OOV/EM 实质恶化，或成本增长与收益不成比例；应保留负面结果，
把 SC-CL 定位为候选排序工具而非无损改进。

## 当前阻塞和可接受表述

当前阻塞不是 LaTeX 编译，而是科学证据仍不完整：缺少同协议完整基线、FB 当前 SC-CL
bank、跨数据集转换和五种子结果。当前可安全写：

- “实现了一个 train-only、冻结式 SC-CL sampler 契约”；
- “论文预先登记了数据集、对照、消融、审计和统计协议”；
- “NELL 单种子结果显示一次候选排序观察，但不代表稳定性或跨数据集泛化”。

当前不可写：

- “SC-CL 在多种子和匹配对照下稳定提升 MRR/EM”；
- “SC-CL 已在七个数据集上验证”；
- “候选一定是真实负例”；
- “结果具有统计显著性”。

## 审阅结论

设计层面已经覆盖主要拒稿路径，但证据层面仍是 partial-experiment。NELL 的 SC-CL
主路径已有一份完整运行，仍必须补齐 SC_MAIN_CORE_NELL 的公平对照、SC_MAIN_CORE_FB、
SC_CTRL_FREQMATCH、SC_AUDIT_LEAK 的 FB 部分和 SC_STAT_SEEDS，再决定是否把标准迁移和
CoDEx 结果升级为主文结论。

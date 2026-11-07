# FB同模型完整重排完成

三组均为相同10,000题、原十个测试选项、相同评分/clip20和原选项顺序破平局；全部候选分数与答案token数保留。它们复用已有模型，没有新训练。

| 方法 | H@1 | MRR | H@3 |
|---|---:|---:|---:|
| Full SC-CL-9 | 94.85% | 97.249940% | 99.75% |
| Full配额匹配随机-9 | 94.72% | 97.151651% | 99.66% |
| Uniform Random-9 | 93.68% | 96.572242% | 99.56% |

Full−配额匹配H@1为+0.13pp，净13题（纠错170、退化157）；MRR为+0.098290pp。Full−Uniform的H@1为+1.17pp，MRR为+0.677698pp。排序上也观察到Full与配额匹配接近，尚不足以支持明确的独有联合选择收益。单seed和Full预选配置继续保留，不将小正差写成统计显著或等效结论。

同模型直接解码EM为89.90/89.83/86.48%，与重排H@1是不同答案选择规则，不把重排相对自身解码的差写成训练收益。这里是十选一MRR，不是全关系或全实体filtered link prediction MRR；原直接解码近零及低频负结果不能删除。

原始300,000个分数、30,000行排名、提示token SHA、token长度、冻结权重身份和日志均在evidence_index所指Linux目录。metrics、paired_ranks_and_direct、rank_distribution、frequency_slice_metrics和各配对CSV供后续绘图。completion_audit_20261107.json记录新一轮流式分数/排名复核及前序完成审计范围。

映射：SC_FB_MATCHED_RERANK / experiments / C2 / claim-carrying。保持C2收窄结论；K敏感性由独立队列继续，不依据本组三行结果更换评分规则或模型。

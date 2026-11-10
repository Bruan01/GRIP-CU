# FB独立训练边频次比例采样对照

## 结果

该固定配置单种子实验已完成，seed=2026，测试集10,000题。采样器仅使用训练图关系边数，按比例逐次无放回抽取9个合法负关系，指数为1；不读取Full候选、B1分数或结构画像，也不继承Full逐题频率桶配额。

| 候选来源 | 直接解码EM | 原十选项H@1 | MRR | H@3 |
|---|---:|---:|---:|---:|
| Full SC-CL-9 | 89.90% | 94.85% | 97.2499% | 99.75% |
| Full配额匹配随机-9 | 89.83% | 94.72% | 97.1517% | 99.66% |
| 均匀随机-9 | 86.48% | 93.68% | 96.5722% | 99.56% |
| 独立频次比例随机-9 | 70.57% | 87.01% | 92.1324% | 96.72% |

独立频次组实际负候选边缘分布到训练边分布的JS散度为0.003215（Full为0.2715）；训练边频率最高的24个关系占其负候选槽位53.10%，Full为11.16%。这说明该bank的全局频率质量接近原始训练边分布，但并未带来较高的任务分数。该分布已在论文FB采样分布附图中展示。

Full相对独立频次比例组的直接EM高19.33个百分点（纠错2,251题、退化318题），H@1高7.84个百分点，MRR高5.1175个百分点。该组直接EM还低于均匀随机15.91个百分点。它说明指数1的全局训练边频次采样不足以在本固定配置下复现Full；不能据此断定其他频率采样方法失效，也不能把差异归因于某个结构信号的因果作用。

## 实验范围与限制

- 共用Full的Stage~1、12,136条训练QA、3,109条候选损失资格、合法池、损失和优化配置；$K=9$。
- 实际训练32步并保持120步学习率调度；Full配置此前按验证集选定。本组没有独立HPO或重复种子。
- 测试使用同一10,000题；排序评价对同一原始十选项打分，不是全图 filtered MRR。
- 所有差异是单个固定配置下的观测结果，不提供跨种子不确定性或等效性结论。共享464条train-pair QA修复及历史测试暴露仍适用。

## 可追溯证据

- [固定实验计划](../../../08_experiments/Hard_Negative_Contrastive_GRIP/docs/fb_frequency_only_control/PLAN.md)
- [运行身份与完成状态](RUNNING.json)；[运行证据索引](evidence_index.json)
- [构库审核](preparation_audit.json)；[训练关系频次](relation_frequencies.csv)
- [汇总结果](result.json)；[逐方法指标](metrics.csv)；[同题多方法配对](paired_all_methods.csv)
- [Full对独立频次组直接解码配对](paired_full_minus_frequency_only_direct.csv)及[排序配对](paired_full_minus_frequency_only_ranking.csv)
- [已选候选及其来源](selected_candidates.csv)
- 外部运行目录：`/home/kieran/grip-cu-runs/fb_frequency_only_control_20261110_seed2026`。论文表源SHA256记录于`../table_sources.json`。

论文映射：`SC_CTRL_FREQONLY_FB`，`section_id=experiments`、`paper_role=claim-carrying`、`claim_links=[C2]`；已纳入候选来源主表、配对表、同列表重排表与附录复现设置。

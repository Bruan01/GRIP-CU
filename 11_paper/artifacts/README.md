# 实验证据索引

本目录保存实测结果与可追溯证据。数值与状态优先使用同运行的逐题数据、结果JSON和审计；历史计划与正文不能覆盖已完成结果。

| 研究问题 | 证据入口 |
|---|---|
| 三数据集获选模型直接解码 | [NELL](nell23k_single_seed_test_20261102.json)、[CoDEx-M](codexm_single_seed_test_20261103.json)、[FB](fb15k237_2_single_seed_test_20261104.json) |
| FB均匀负采样控制 | [结果](fb_random9_control_seed2026/result.json) |
| FB两个单信号控制 | [汇总](fb_component_ablation_seed2026/result.json)、[Profile](fb_profile_only_control_seed2026/result.json)、[Violation](fb_violation_only_control_seed2026/result.json) |
| FB Full诱导频率配额匹配 | [结果与解释](fb_frequency_matched_control_seed2026/INTERPRETATION.md) |
| FB三来源完整同候选重排 | [结果与解释](fb_matched_rerank_seed2026/INTERPRETATION.md) |
| FB频率分层及关系等权宏平均 | [解释](fb_frequency_slices_seed2026/INTERPRETATION.md)、[逐题](fb_frequency_slices_seed2026/question_slices.csv)、[逐关系](fb_frequency_slices_seed2026/per_relation.csv) |
| FB训练K3/9/15排序敏感性 | [汇总解释](fb_k_sensitivity_seed2026/INTERPRETATION.md)，K3/K15单独目录保留训练曲线与配置 |
| NELL三来源控制，已完成 | [结果](nell23k_source_controls_seed2026/result.json)、[逐题](nell23k_source_controls_seed2026/paired_all_methods.csv) |
| WN三来源控制，已完成、探索性 | [结果及限制](wn18rr_source_controls_seed2026/result.json)、[逐题](wn18rr_source_controls_seed2026/paired_all_methods.csv) |
| NELL后续未改善结果 | [续训测试](nell23k_refinement_test_20261104.json)、[第二轮验证](nell23k_round2_no_go_20261105.json) |
| 最新PDF与实验一致性审查 | [修订报告](../review/SC-CL_sota_revision_audit_20261110.md)、[独立核验JSON](../review/sota_evidence_audit_20261110.json) |

NELL控制的早期检查点丢失说明保留在对应`checkpoint_retention.json`；WN实体对核验缺口和历史B1差异保留在结果限制中。配额匹配组继承Full逐题选桶配方，不是独立的一般频率采样方法。

原始用户图稿保留为`方法架构图.jpg`与`结构引导与候选级对比优化流程图.svg`；当前仓库正文使用`main/figures/sccl_mechanism.tex`。

本目录仍有历史证据账本与检查记录，它们表示当时状态。历次写作文本、审阅意见和截图已移至[历史档案](../../12_ai_logs/paper_history/20261110/README.md)。本次清理映射及SHA256见[cleanup_20261110.json](cleanup_20261110.json)，此前清理见[cleanup_20261108.json](cleanup_20261108.json)。

# 频率控制完成后的论文表述

## 可用于结果段

在FB15K237-2固定配置、seed2026的完整测试中，频率桶匹配随机候选获得89.83%的直接解码EM，相比均匀随机的86.48%提高3.35个百分点，与Full SC-CL的89.90%仅相差0.07个百分点（净7/10,000题）。Full独有正确348题，反向341题；689题发生判定变化。因此，总体EM接近不意味着模型逐题行为相同。当前结果未能排除候选频率分布对先前均匀随机对照差异的解释，不足以支持联合选择在其诱导频率桶分布之外具有明确额外收益的主张。

## 必须一起说明

匹配的是Full训练候选库逐题逐槽的`floor(log2(train_edge_count))`桶配额，同桶内均匀无放回抽样，保留相同train-known合法池。全部3,109题精确匹配，零回退；27,981槽中82槽因小池被迫选定。配额本身来自Full，已经携带Full策略选择的信息，并非独立于Full的一般频率基线；桶内具体频次、B1困难度未匹配。本轮只比较完整测试的直接解码EM，不能代替候选重排/MRR的对照结论。不能据此断言频率是全部原因、结构无效或两模型统计等效。单seed及此前Full选定配置不提供跨seed稳定性或各方法最优性能比较。

同Stage1、12,136条训练QA、3,109候选资格、K=9、InfoNCE、LoRA与优化器；32优化步并保持120步调度。第32步在测试前固定，各组件只做一次完整测试；新组无完整验证、HPO或重复种子。共享464条train-pair QA修复和历史测试暴露。原有Full−Violation+2.81pp与Full−Profile+5.01pp仍是观测差异，但不能据此排除频率中介或宣称普适协同。

## 写作映射与下一步

- selected_outline_ref: `11_paper/paper_outline.md`
- exp_id/item_id: SC_CTRL_FREQMATCH；section_id: experiments；claim_links: C2
- paper_role: claim-carrying；analysis_role: contradiction check；target_display: 主文五方法候选来源表
- research_question: 保持Full诱导的频率桶配额后随机选关系，能否得到接近的完整测试EM？
- experimental_design: 同合法池、同频率桶配额随机候选替换；固定优化和评测协议。
- evaluation_summary: 频率桶匹配89.83%，Full89.90%，Uniform86.48%；Full−匹配+0.07pp、匹配−Uniform+3.35pp。
- claim_update: 收窄C2；未排除频率替代解释，不再以均匀随机差异独立支持联合信号优势。
- comparison_baselines: Full、Uniform Random；Profile与Violation保留于五方法表。
- todo_items: 后续正文加入此近零控制并同步收窄机制叙事；原实测值与负结果不删除。
- next_route: write；停止本轮GPU实验，不为追求正差异重采bank、调桶、换seed或延长训练。

## 绘图与过程证据

同目录`metrics.csv`、`paired_five_methods.csv`、两组配对差CSV、`diagnostics.csv`、`frequency_bin_distribution.csv`、`relation_frequencies.csv`和`training_history.csv`可用于后续绘图。逐候选55,962行的原始频次/结构/B1诊断保留在Linux的`bank/selected_candidates.csv`，完整源路径见`evidence_index.json`。四个模型检查点及optimizer/scheduler/RNG、原始response、冻结代码和日志均保留，借用的Random源输入也须保留。

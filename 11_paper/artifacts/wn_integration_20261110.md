# WN18RR结果写入策略与证据映射

## 写作前的修订范围

- 完成结果未写入：WN三来源完整直接解码与重排已完成；动作是增加正文跨图对照表和附录条件说明。
- 稿件映射缺口：仓库正文仍未纳入NELL三来源，无法与WN/FB并列解释；动作是同表完整保留NELL三行及正负差，不启动实验。
- 不受支持的概括：FB的配额近零差不能推广至WN；动作是将讨论、摘要与结论改为依赖数据集和答案选择规则的条件观察。
- 状态漂移：WN/NELL在实验矩阵、提纲、章节计划及结果状态中仍被列为待运行；动作是以现有result.json更新当前状态，历史进度单独标明。
- 证据缺口不补写成完成：多种子、各方法独立调参、WN的55条实体对核验与B1复现差异仍未解决；放入附录限制，WN保持探索性身份。
- 引用范围：沿用既有GRIP引用说明WN任务与原文参考，不新增未经核验的参考文献，也不将本轮作为新颖性文献审查。

## 数值与页面映射

来源为`nell23k_source_controls_seed2026/result.json`、`wn18rr_source_controls_seed2026/result.json`及各自`paired_all_methods.csv`。逐题重算复用`review/audit_sota_evidence_20261110.py`，生成表加入`main/tables/build_result_tables.py`并登记来源哈希。

- 正文`sec:cross-dataset-sources`与`tab:cross-dataset-sources`：NELL与WN各三方法的EM/H@1/MRR/H@3，全量同图同选项。
- 附录`sec:wn-exploratory`：WN训练/验证选模、历史Full复用、11选项、55条未核实实体对、小合法池及B1复现限制。
- 附录`sec:cross-source-pairs`与`tab:cross-source-pairs`：两数据集三配对方向下的直接/排序纠错、退化及净差；NELL检查点保留例外。
- 摘要/引言/讨论/结论：保留三主数据集配置比较，WN作为第四数据集的探索性来源对照，不把它并入已修复B1的主效果结论。

## 验证范围

完成后核对生成表与JSON/逐题数据、执行CPU审计、从零编译并检查引用和表格溢出，渲染检查新增正文表与附录表。PDF交付为`main/main.pdf`；用户提供的13页PDF保留，不覆盖。未启动训练或GPU推理。

# 文献启发的采样诊断图（三个有训练对照的数据集）

直接阅读[七页完整PDF](sampling_diagnostics_all.pdf)，或[全部预览](overview.png)。[数据解读与限制](../../../artifacts/sampling_diagnostics_seed2026/README.md)记录了CoDEx-M从论文图中排除的原因，[调研笔记](../../../review/sampling_plot_literature_20261110.md)说明各图借鉴什么。

| 图 | 回答的问题 | 建议位置 |
|---|---|---|
| [三数据集分布](fig01_selection_distributions.pdf) | Full是否偏向题内更困难、结构更相似的合法负例？ | 主文分析，双栏宽 |
| [联合富集](fig02_joint_selection_lift.pdf) | 哪些结构×困难度区域相对合法池均匀参考被增选？ | 主文或附录，双栏宽 |
| [原始与采样边际](fig03_relation_marginals.pdf) | 采样关系概率是否沿原始训练边概率的相等线？ | 主文或附录；直接回应原始分布问题 |
| [FB成分指纹](fig04_fb_component_fingerprints.pdf) | 已完成的Profile/Violation来源是否改变了预期信号？ | 组件分析，双栏宽 |
| [NELL](diagnostics_nell23k.pdf) | 原始分数、联合富集和逐题变化 | 附录 |
| [FB](diagnostics_fb15k237_2.pdf) | 同上；实际来源对齐 | 附录 |
| [WN](diagnostics_wn18rr.pdf) | 同上；历史池、55个未核验实体对 | 附录 |

每张提供同名PDF、SVG、PNG；SVG保留可编辑文字，PDF使用TrueType文字。[LaTeX插入片段](paper_figures.tex)含完整图注。CoDEx-M没有已训练的Uniform或Quota对照，因此不纳入比较图或正文诊断结论；没有为其补跑训练。前一轮边际JS图包继续保留。

中文图注要点：每题等权，K=9、seed2026。题内百分位越大表示相对更相似或更困难，绝对数值不代表真假概率。联合图的参照为逐题合法池上的解析Uniform；本图版仅展示NELL23K、FB15K237-2和WN18RR。WN保留历史范围。配对ECDF每个观测是一道训练题，正值表示Full的平均困难度更高；它不是训练种子重复分布。

范围说明：旧审阅摘要中的CoDEx-M配对差值−0.030无法由此图包的同定义来源复现；当前冻结CSV对Full减解析Uniform的题内困难度百分位给出均值+0.052。由于CoDEx-M没有已训练的逐题对照，当前图版不展示其诊断，也不追加训练来追查该差值。

主文建议表述：

> 在具有已训练均匀与配额对照的三个冻结候选库上，Full在NELL23K和FB15K237-2更明显地向较高题内结构相似度与B1困难度移动，WN18RR的移动较弱。FB单信号对照进一步显示，结构画像与B1反馈主要改变各自对应的采样属性。这些结果描述采样行为与设计方向的一致性；结合配额匹配控制，它们不足以将下游性能差异完全归因于边际频率或平均困难度。

重绘：在仓库根目录执行guardenv Python `11_paper/main/figures/sampling_diagnostics_seed2026/build_figures.py`。无需GPU或原始评分。图形QA、源数据哈希在`qa_report.json`；渲染检查记录在`render_review.json`。

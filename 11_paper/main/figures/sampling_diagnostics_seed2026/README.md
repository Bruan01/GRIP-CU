# 文献启发的采样诊断图

直接阅读[八页完整PDF](sampling_diagnostics_all.pdf)，或[全部预览](overview.png)。[数据解读与限制](../../../artifacts/sampling_diagnostics_seed2026/README.md)包含可用于写作的数值，[调研笔记](../../../review/sampling_plot_literature_20261110.md)说明各图借鉴什么。

| 图 | 回答的问题 | 建议位置 |
|---|---|---|
| [四数据集分布](fig01_selection_distributions.pdf) | Full是否偏向题内更困难、结构更相似的合法负例？ | 主文分析，双栏宽 |
| [联合富集](fig02_joint_selection_lift.pdf) | 哪些结构×困难度区域相对合法池均匀参考被增选？ | 主文或附录，双栏宽 |
| [原始与采样边际](fig03_relation_marginals.pdf) | 采样关系概率是否沿原始训练边概率的相等线？ | 附录；直接回应原始分布问题 |
| [FB成分指纹](fig04_fb_component_fingerprints.pdf) | 已完成的Profile/Violation来源是否改变了预期信号？ | 组件分析，双栏宽 |
| [NELL](diagnostics_nell23k.pdf) | 原始分数、联合富集和逐题变化 | 附录 |
| [CoDEx-M](diagnostics_codexm.pdf) | 同上；只有解析Uniform | 附录 |
| [FB](diagnostics_fb15k237_2.pdf) | 同上；实际三来源对齐 | 附录 |
| [WN](diagnostics_wn18rr.pdf) | 同上；历史池、55个未核验实体对 | 附录 |

每张提供同名PDF、SVG、PNG；SVG保留可编辑文字，PDF使用TrueType文字。线型、标记和数据标签辅助灰度阅读。[LaTeX插入片段](paper_figures.tex)含完整英文图注，尚未插入正文，两个既有全文PDF均保持原状。前一轮边际JS图包继续保留。

中文图注要点：每题等权，K=9、seed2026。题内百分位越大表示相对更相似或更困难，绝对数值不代表真假概率。联合图的参照始终为精确解析Uniform；CoDEx无已训练对应对照。WN保留历史范围。配对ECDF每个观测是一道训练题，正值表示Full的平均困难度更高；它不是训练种子重复分布。

主文建议表述：

> 在冻结候选库上，Full在NELL23K、CoDEx-M和FB15K237-2中呈现向较高题内结构相似度与B1困难度移动的采样分布，而WN18RR的移动较弱。FB单信号对照进一步显示，结构画像与B1反馈主要改变各自对应的采样属性。这些结果验证了采样行为与设计方向的一致性；结合配额匹配控制，它们不足以将下游性能差异完全归因于边际频率或平均困难度。

重绘：在仓库根目录执行guardenv Python `11_paper/main/figures/sampling_diagnostics_seed2026/build_figures.py`。无需GPU或原始评分。图形QA、源数据哈希在`qa_report.json`；本轮检查记录在`render_review.json`，文件完整性在数据包`package_check.json`。

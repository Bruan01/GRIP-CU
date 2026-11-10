# 四数据集采样分布图

全部图来自实际seed2026冻结候选库与训练图，未启动模型、训练或新候选抽样。

先看[六页完整PDF](sampling_distributions_all.pdf)、[四数据集累积分布](fig01_cumulative_four_datasets.pdf)与[分布距离总图](fig02_distribution_distance.pdf)。

| 数据集独立四面板 | PDF | 可编辑SVG |
|---|---|---|
| NELL23K | [PDF](distribution_nell23k.pdf) | [SVG](distribution_nell23k.svg) |
| CoDEx-M | [PDF](distribution_codexm.pdf) | [SVG](distribution_codexm.svg) |
| FB15K237-2 | [PDF](distribution_fb15k237_2.pdf) | [SVG](distribution_fb15k237_2.svg) |
| WN18RR | [PDF](distribution_wn18rr.pdf) | [SVG](distribution_wn18rr.svg) |

每张图另有300dpi PNG。独立图均为183mm宽；总览见[彩色](overview_four_datasets.png)与[灰度检查](overview_four_datasets_grayscale.png)。热图纵轴括号标出合格QA数；强差异以正负符号辅助识别，符号阈值仅为显示规则。灰色热图行无对应gold训练问题，不是观测到的零变化。

结论：在四图的原有分析中，Full比同图均匀参考更接近训练图关系边分布，但未复刻原图的高频集中程度。2026-11-11完成的FB独立频次比例控制已叠加到FB四面板图的a、b、d：实际负候选分布到训练边分布的JS散度为0.003215，Full为0.2715；训练频次最高24个关系占独立频次组负候选的53.10%，占Full的11.16%。这一边缘分布的吻合来自按训练边数比例抽样，不代表更好的关系预测；该组直接EM为70.57%，见候选来源结果表。c面板仍只表示Full减合法池Uniform的逐题条件差异，其他三个数据集图不变。

CoDEx-M没有已训练的匹配控制；其Uniform始终标为解析期望。FB的237个原始关系ID在项目既有文本映射后为233个训练标签。WN包含55条历史实体对未核实训练QA，核验限制与子集敏感性见[分析报告](../../../artifacts/sampling_distribution_seed2026/README.md)。

重绘（只需随仓库保存的干净CSV，不依赖原始大文件）：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python 11_paper/main/figures/sampling_distribution_seed2026/build_figures.py
```

[主文插入片段](paper_figures.tex)与[附录四图片段](appendix_figures.tex)提供完整图注；可从论文`main/`目录使用`\input{figures/sampling_distribution_seed2026/paper_figures.tex}`，附录另引`appendix_figures.tex`。四张数据集图已由`main/sections/appendix.tex`插入附录；本轮重新编译全文PDF。

来源、公式与四图原分析见[分析目录](../../../artifacts/sampling_distribution_seed2026/README.md)，导出检查见[qa_report.json](qa_report.json)。新增PPS曲线和频率桶由`build_figures.py`读取[实际已选候选](../../../artifacts/fb_frequency_only_control_seed2026/selected_candidates.csv)，并与FB训练边计数逐关系核对；JS由脚本重算。FB图以附录插图标签`fig:sampling-fb`集成，surface class为appendix。

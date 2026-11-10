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

结论：Full比同图均匀参考更接近训练图关系边分布，但未复刻原图的高频集中程度。配额匹配控制与Full逐题桶组成相同，不能根据粗粒度频率图判断关系身份或预测性能。

CoDEx-M没有已训练的匹配控制；其Uniform始终标为解析期望。FB的237个原始关系ID在项目既有文本映射后为233个训练标签。WN包含55条历史实体对未核实训练QA，核验限制与子集敏感性见[分析报告](../../../artifacts/sampling_distribution_seed2026/README.md)。

重绘（只需随仓库保存的干净CSV，不依赖原始大文件）：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python 11_paper/main/figures/sampling_distribution_seed2026/build_figures.py
```

[主文插入片段](paper_figures.tex)与[附录四图片段](appendix_figures.tex)提供完整图注；可从论文`main/`目录使用`\input{figures/sampling_distribution_seed2026/paper_figures.tex}`，附录另引`appendix_figures.tex`。本任务提供图表包和插入片段，未自动增加正文篇幅或覆盖现有全文PDF。

来源、公式与结果解释见[分析目录](../../../artifacts/sampling_distribution_seed2026/README.md)，导出检查见[qa_report.json](qa_report.json)。

# SC-CL 论文图表包（2026-11-03 数据快照）

使用 **nature-figure skill / Python**，以真实预测重新核验并生成三张结果图。默认重建仅使用随包 CSV，执行过程不加载模型、不启动训练，也不依赖本机原始推理文件。

快速查看：[三页 PDF](figures_overview.pdf) · [PNG 总览](figures_overview.png) · [灰度检查](figures_overview_grayscale.png)。每张图另有可编辑文字 SVG、嵌入 TrueType 字体的 PDF、600 dpi LZW TIFF、300 dpi PNG。

| 图 | 内容 | 论文中的用途 |
|---|---|---|
| [图 1](fig01_direct_decoding.pdf) | NELL23K / CoDEx-M 全量直接解码及净增幅 | 最新本地结果的主效果图 |
| [图 2](fig02_paired_outcomes.pdf) | 匹配题目的纠错、退化、共同正确及共同错误 | 解释净增益的逐题来源 |
| [图 3](fig03_historical_ranking.pdf) | 历史 NELL23K 的前 200 题配对排序与 SC-CL 全量排序 | 与当前正文候选重排叙事对应的支持图 |

## 核验后的数值

| 最新直接解码 | 测试题数 | 匹配 B1 EM | Stage-2 SC-CL EM | 差异（百分点） | 净增正确题 |
|---|---:|---:|---:|---:|---:|
| NELL23K | 4,943 | 84.58% | 87.80% | +3.22 | +159 |
| CoDEx-M | 10,311 | 94.34% | 96.58% | +2.24 | +231 |

NELL23K：纠错 321、退化 162、共同正确 4,019、共同错误 441。CoDEx-M：纠错 301、退化 70、共同正确 9,657、共同错误 283。

| 历史 NELL23K 排序范围 | 模型 | H@1 | MRR | H@3 |
|---|---|---:|---:|---:|
| 测试顺序前 200 题 | B1 | 88.50% | 93.36% | 97.50% |
| 同候选前 200 题 | SC-CL | 91.00% | 94.96% | 99.50% |
| 全量 4,943 题 | SC-CL | 91.52% | 95.27% | 99.07% |

图 3 的历史模型、历史基线和候选评分路径，与图 1–2 的最新直接解码比较不同。全量 B1 重排尚缺，图中没有补造全量 B1。当前正文主要沿用较早的排序实验，本任务提供独立图表包和插入片段，正文叙事仍需随实验进展统一。

## 图注

**图 1｜单种子本地协议下的完整测试直接解码。** a，匹配 B1 和验证选定的 Stage-2 SC-CL 配置的 Exact Match；NELL23K 的正确数为 4,181/4,943 与 4,340/4,943，CoDEx-M 为 9,727/10,311 与 9,958/10,311。b，两模型在同一测试队列上的 EM 净差及净增正确题数。训练种子均为 2026，每数据集一个获选模型。柱高为该模型完整测试的观测比例；无跨种子重复，因此不绘制标准差、置信区间或显著性标记。比较涉及 Stage-2 调参，不能单独归因为候选损失或结构采样。

**图 2｜直接解码的逐题配对变化。** 行为匹配 B1 的正确性，列为 Stage-2 SC-CL 的正确性。每格显示题数及其占该数据集全测试队列的百分比。Rescue 指 B1 错误而 SC-CL 正确，regression 指相反变化。各数据集的净增题数为两者之差。矩阵为描述性统计，不是跨种子不确定性估计。

**图 3｜历史 NELL23K 候选重排结果。** a，按测试文件顺序取前 200 题，在相同十个候选及候选顺序下，以 teacher-forced continuation score 比较 B1 与 SC-CL。b，SC-CL 在全量 4,943 题上的结果。H@k 为目标关系 rank 不大于 k 的比例，MRR 为目标 rank 倒数的均值；均以百分数表示。全量 B1 重排未完成；两模型训练预算有差异；训练种子为 2026。该图不能与最新直接解码结果交叉推导方法收益。

## 适用边界

- 这些是单种子实测，尚无 matched Random-9 等消融或多种子结论。
- NELL 此前有探索性测试暴露，任务生成使用过验证 few-shot；方法和控制的 Adam epsilon 不同。
- CoDEx-M 使用 5-epoch 调度并选中第 32 步，与原论文 10-epoch 设置不同，训练 microbatch 亦不同；提示与候选协议仍待审计。
- 本包不将原论文数值作为本地配对控制，不宣称严格复现或统计显著；FB 当前方法测试未完成，WN18RR 仍属探索性记录。
- 本包为论文工作图；尚未核对特定期刊最新投稿规范。

## 复现与插入论文

在仓库根目录执行：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python \
  11_paper/main/figures/python_results_20261103/build_figures.py
```

只有需要重新审计原始预测时才加 `--refresh-source`。依赖为 matplotlib、numpy、Pillow、pypdf；均已在 guardenv 中验证可用。所有绘制、预览和导出使用 Python。

从论文 `main/` 目录编译时，可在选定位置加入：

```latex
\input{figures/python_results_20261103/paper_figures.tex}
```

[插入片段](paper_figures.tex)包含三张图、中文图注及一张最新结果表，需要已有的 `graphicx` 和 `booktabs` 宏包。请先统一正文中较早结果与最新直接解码的叙事；本包不自动改变现有正文。

## 数据与质量检查

- [direct_pairs.csv](source_data/direct_pairs.csv)：15,254 个匿名配对问题，两个模型的正确标记，共 30,508 个标记。
- [ranking_records.csv](source_data/ranking_records.csv)：4,943 条 SC-CL gold rank 及前 200 题 B1 gold rank。空 B1 rank 表示未完成，不是 0。
- [direct_metrics.csv](source_data/direct_metrics.csv)、[paired_outcomes.csv](source_data/paired_outcomes.csv)、[ranking_metrics.csv](source_data/ranking_metrics.csv)：由逐题 CSV 重算的图中数值。
- [provenance.json](source_data/provenance.json)：来源证据标识与 SHA-256，不复制绝对机器路径、提示文本或模型权重。最新预测散列均与已有审计记录一致；历史排序原始文件散列本次记录。
- [qa_report.json](qa_report.json)：CSV 和导出文件散列、物理尺寸、600 dpi TIFF、SVG 文字节点、PDF 文字提取和字体检查。

已检查：问题标识无重复；同队列题目、提示、目标和测试划分对齐；30,508 个 EM 标记使用项目原始 normalization 函数重算；配对四分类与既有审计一致；5,143 条排序记录从候选分数重算 gold rank；前 200 题候选顺序对齐；净变化恒等式成立；三页 PDF 完整；SVG 文字保持可编辑；PDF 可提取文字且无 Type3 字体；所有文字均在画布内。

已人工查看完整 PNG 总览和灰度预览，未发现明显遮挡、裁切或不可辨识配色。颜色同时辅以斜线填充、符号和文字，图 2 采用同尺寸分类单元格而非靠面积表达题数。导出的 SVG 以 XML、PDF 以文字和页面结构检查；未使用独立 PDF/SVG 光栅化器复核，因此特定外部编辑器的字体替换仍需在排版时确认。

另已验证：默认命令仅凭随包 CSV 重建成功，五份 CSV 的 SHA-256 均保持一致；LaTeX 插入片段经两轮 XeLaTeX 独立编译生成两页文档，无 overfull box。编译检查使用临时文档，未改写或重编译原论文正文。

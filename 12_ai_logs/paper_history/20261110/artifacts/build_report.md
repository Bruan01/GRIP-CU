# 当前论文构建与核验（2026-11-08）

正式入口：`11_paper/main/main.tex`；正式PDF：`11_paper/main/main.pdf`（14页）。`main/build/`仅为可再生编译缓存。

- 使用guardenv Python生成七张有来源的正文/附录表，输入hash见`table_sources.json`。
- XeLaTeX + BibTeX经latexmk编译通过；无未解析引用/文献、缺字或Overfull box。
- 中文smallcaps/粗体/斜体存在字体替代，少量Underfull box仍保留；不影响所检查页面的阅读。
- 首页面和分析/讨论页已渲染检查，表格清晰且无重叠。
- 从两份10,000题CSV重算三来源及三个K点的H@1/MRR/H@3和全部配对纠错/退化，与正式结果JSON一致。K9复用，不将其重复计作独立训练。
- 21个LaTeX依赖、36个label和16个实际引用key检查通过；57个原有受版本控制的图形文件全部保留。
- `git diff --check -- 11_paper`通过。未新增训练、种子或HPO，未重跑GPU原始评分；全候选分数审计沿用已完成审计，本轮重新计算的是逐题rank导出。

机器核验记录：`manuscript_validation_20261108.json`。删除清单与SHA：`cleanup_20261108.json`；删除的历史稿可从Git恢复。

当前证据仍不支持跨seed稳定性、统计等效、独有结构机制或全局最优K；编译通过不等于投稿就绪。

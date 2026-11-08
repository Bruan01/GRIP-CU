# SC-CL 论文工作区

正式正文：[main/main.tex](main/main.tex)。当前全文PDF：[main/main.pdf](main/main.pdf)。
目标为二区期刊，具体期刊待选；ICLR 2026模板仅用于工作稿排版。

## 叙事与结果

论文研究固定实体对上的关系候选学习：结构画像与冻结B1反馈构建可追溯候选库，来源对照进一步区分候选身份与其频率组成的作用。

主文依次呈现三数据集完整直接解码、FB五来源固定配置对照、三来源同题重排；独立分析章节解释频率分层和训练负例K敏感性。FB Full/均匀随机/Full配额匹配的直接EM为89.90/86.48/89.83%，重排H@1为94.85/93.68/94.72%。K3/9/15的H@1为95.28/94.85/95.03%。保留Profile负结果、低频退化与近零差异，不认定独有结构机制、统计等效、最优K或跨seed稳定性。

当前仍为单种子中期稿。NELL/WN候选来源实验运行状态不自动视作完成证据；多种子、跨图公平对照和开放世界真值风险审计尚需补齐。历史NELL排序、FB i0及WN探索边界独立放入附录。CLEGR图级QA不纳入本文。

## 文件职责

- `paper_outline.md`：叙事、主张及证据门；`section_plan.md`：章节与显示职责。
- `paper_experiment_matrix.md/json`：实验矩阵；`results_status.md`：运行与结果状态。
- `main/sections/`：唯一正式章节；`main/references.bib`：正式文献库。
- `main/tables/`：从实测JSON重建的正文表；`main/figures/`：图表及生成脚本。
- `artifacts/`：结果JSON/CSV、审计、证据账本及写作核验记录；`review/`：历史审阅依据。
- `template/iclr2026/`：编译所需样式、数学命令与参考文献样式。

2026-11-08删除重复教学正文、修订前快照正文和旧全文PDF；图表保留，历史正文可从Git恢复。精确清单见`artifacts/cleanup_20261108.json`。旧审计中的路径是历史记录，不是当前入口。

## 重建

仓库根目录运行：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python 11_paper/main/tables/build_result_tables.py
cd 11_paper/main
BSTINPUTS=../template/iclr2026: latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

需要XeLaTeX和正文指定的中文字体。编译产物位于`main/build/main.pdf`，核验后复制到正式阅读版`main/main.pdf`；七张表的输入hash写入`artifacts/table_sources.json`。本轮修订与验证记录见`artifacts/writing_revision_20261108.md`。

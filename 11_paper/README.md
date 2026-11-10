# SC-CL 论文工作区

## 阅读与修改入口

- [用户提供的新版论文 PDF](SC-CL_sota_paper.pdf)：本轮审查对象，13页；尚未获得其对应LaTeX源文件。
- [最新修订审查](review/SC-CL_sota_revision_audit_20261110.md)：逐页问题、实验遗漏、来源与可直接使用的替换文字。
- [仓库 LaTeX 正文](main/main.tex)及[对应 PDF](main/main.pdf)：现有可编辑版本，与上面的新版PDF不同，已于2026-11-10加入NELL/WN三来源正文结果与附录核验说明。
- [实验证据索引](artifacts/README.md)：按实验查找结果、逐题CSV和审计。
- [Gemini交接包](gemini_handoff/00_README.md)：2026-11-09静态快照；其WN“未完成”状态已过时，最新结果见证据索引和修订审查。

目标为二区期刊，具体期刊待选；ICLR 2026模板用于工作稿排版。

## 当前结果

论文研究固定实体对上的关系候选学习：结构画像与冻结B1反馈构建固定候选库，再区分候选身份与频率组成的影响。

三个主数据集的获选配置已完成直接解码评测。FB五来源控制、三来源重排、频率分层和K3/9/15排序敏感性均已完成；NELL与WN三来源控制也已完成，不能继续按旧进度文字视为待运行。用户提供的PDF已包含NELL控制但缺少WN；本仓库LaTeX及其PDF已加入两图完整来源结果。WN仍是历史候选池条件下的探索性证据，需保留实体对核验和B1复现限制。

所有结果为单种子证据；保留Profile-only负结果、配额匹配近零差异、低频退化和跨协议反向差异。既有调参与开发历史的限制见修订审查。CLEGR图级QA不纳入本文。

## 目录职责

| 路径 | 内容 |
|---|---|
| `main/` | 可编辑正文、章节、引用、结果表与TikZ机制图 |
| `main/figures/python_results_20261103/` | 可独立重建的实测图表包及来源CSV；固定历史数据快照 |
| `artifacts/` | 实验结果、逐题数据、来源审计、原始用户图稿与清理清单 |
| `review/` | 最新PDF修订审查、核验脚本和机器可读结果 |
| `gemini_handoff/` | 可携带的历史交接快照 |
| `template/iclr2026/` | 编译需要的样式、数学命令和参考文献样式 |
| 根目录规划文件 | `paper_outline.md`、`section_plan.md`、实验矩阵、`results_status.md`及`EXPERIMENT_ROADMAP.md`；包含历史时点，状态以实验结果和最新审查为准 |

2026-11-10已清理未使用图形方案、占位图、教学残留、构建缓存和表格预览。历史写作与审阅记录移至[历史档案](../12_ai_logs/paper_history/20261110/README.md)，移动文件保持原字节。删除路径、原始SHA256及Git恢复基点见[清理清单](artifacts/cleanup_20261110.json)。实测结果、正式正文、两个版本PDF、原始用户SVG/JPG及实测图表包保留。

## 重建

在仓库根目录重建表格：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python 11_paper/main/tables/build_result_tables.py
```

编译仓库LaTeX版本：

```bash
cd 11_paper/main
BSTINPUTS=../template/iclr2026: latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

需要XeLaTeX和正文指定的中文字体。编译缓存位于`main/build/`，不进入Git；确认后再替换`main/main.pdf`。表格来源哈希记录在`artifacts/table_sources.json`。

重算本轮PDF审查所用的结果：

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python 11_paper/review/audit_sota_evidence_20261110.py
```

该脚本只进行CPU核验，不启动训练或推理。

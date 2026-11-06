# 11 Paper：面向开放世界关系候选排序的结构反事实对比学习

本目录是当前论文工作区。论文主线已经固定为 **Structure-Counterfactual Contrastive
Learning (SC-CL，结构反事实对比学习)**，基于原始 GRIP 的候选关系学习阶段，不引入
递归执行器或 RecurrentGRIP 运行时。

## 目录

实验矩阵记录 NELL23K、FB15K237-2、WN18RR 和 CoDEx-M 四个关系候选排序基准，并区分数据转换、task inference 与模型运行状态。CoDEx-M 官方实体替换硬负例不直接作为关系排序候选标签。CLEGR 图级问答不纳入本文。

- `template/iclr2026/`：当前工作稿使用的 ICLR 2026 LaTeX 模板。论文目标为二区期刊；选定具体期刊后需按其投稿模板重排，当前模板不代表投稿 venue。
- `main/main.tex`：可编译的中文论文主文件；章节位于 `main/sections/`。
- `main/references.bib`：论文实际使用的文献库。
- `main/figures/`：机制图、实测结果图及其脚本；旧占位图保留作历史文件。
- `artifacts/`：intake audit、证据账本、实验矩阵和结果状态。
- `review/`：独立的首轮审阅与修订清单。

## 编译

在 WSL 中从本目录运行：

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/11_paper/main
export BSTINPUTS="$(pwd)/../template/iclr2026:${BSTINPUTS:-}"
latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

输出为 `main/build/main.pdf`。XeLaTeX 是必要的，因为正文为中文；本机已确认
`xelatex`、`latexmk`、`xeCJK` 和中文字体可用。

## 结果状态

当前为单种子实测与未完成协议并存的中期稿。NELL23K、CoDEx-M、FB15K237-2的验证选定配置已完成全测试直接解码（87.80%、96.58%、89.90%），主文明确其包含训练调参，不能解释成纯采样收益。WN18RR保留探索性结果及B1复现差异。

FB五方法固定配置对照已完成：Uniform86.48%、Profile84.89%、Violation87.09%、Full配额匹配随机89.83%、Full89.90%。配额来自Full逐题训练频率桶，不是独立通用频率采样器。Full仅高0.07pp，频率替代解释未排除；低频组Full净少17题，按关系等权EM也未优于配额匹配组。全部正、近零和负结果并列呈现。

## 论文叙事主线

采用“可追溯候选设计—配置表现—固定配置候选替换—频率配额解释边界—描述性分层”的顺序。五份证据生成的主文/附录表位于`main/tables/`，可用`build_result_tables.py`重建。历史NELL候选重排及其旧直接解码模型放入附录，与当前模型身份分开；历史FB i0不改名为SC-CL。

新分析只复用已有五模型预测：训练关系频次确定215/689两个边界，所有10,000题均包含，报告三层级及按关系等权平均；无新GPU训练、HPO或种子。原始模型、检查点、优化器/RNG、响应与日志保留Linux，轻量表、脚本及来源索引保存在`artifacts/fb_frequency_slices_seed2026/`。

## 内部证据要求

`results_status.md` 与 `artifacts/evidence_ledger.md` 保留实验状态、对照缺口和证据门。历史 FB i0 与 SC-CL 分别记录；候选重排与直接解码采用不同答案选择规则。显著性、跨种子稳定性、组合信号独立收益及跨图收益仍需对应对照和原始产物支持。

## 本轮写作修订

2026-10-28 已按逐处审查重组正式稿，并更新可编辑矢量流程图。修订记录见 `artifacts/writing_revision_20261028.md`；修改前源文件与 PDF 保存在 `artifacts/before_writing_revision_20261028/`。课堂审稿材料 `teaching/` 保持其独立用途。

2026-11-06修订：把当前完整测试、FB五方法控制与频率切片正式写入摘要、引言、实验、讨论和结论；修订策略、来源映射、构建与检查记录见`artifacts/writing_revision_20261106/`。当前稿仍不提供多种子、统计等效或独有结构机制收益结论。

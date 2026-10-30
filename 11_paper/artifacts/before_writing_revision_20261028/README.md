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

当前是单种子实测与待运行协议并存的中期稿：NELL23K 已有 seed=2026 的 SC-CL
训练、bank、wiring、full candidate rerank 和直接解码；FB15K237-2 只有 B1 与历史 InfoNCE-i0
候选评分，不能当作当前 SC-CL 主结果。WN18RR 已完成一次探索性的全测试直接解码，尚缺完整候选重排与匹配对照；CoDEx-M 方法结果、五种子、同协议对照、
消融和假阴性审计仍按 `results_status.md` 标为待运行。用户另报告一项 CLEGR 异机运行比原论文高
`0.415`；该图级问答结果与本文的关系候选方法不构成可比证据，仅在内部状态中留档，不进入
本文的实验矩阵、结果表或结论。所有论文结果均应保留原始预测、manifest、环境 hash 和训练配置。

## 论文叙事边界

论文范围限定为开放世界知识图谱的关系候选排序。论文只声称：提出一种 train-only、冻结式的结构反事实候选库，并将其接入原始
GRIP 的候选关系 InfoNCE；候选的“反事实”含义是对观测实体对替换关系标签后，
过滤 train-known 关系，不把开放世界中未观测事实视为绝对负例。NELL 的单种子结果
只支持一次运行下的候选排序观察，不支持显著性、稳定性、OOV 修复或通用负采样理论。候选重排 EM/H@1 与模型直接解码 EM 使用不同的答案选择规则，论文分别报告。

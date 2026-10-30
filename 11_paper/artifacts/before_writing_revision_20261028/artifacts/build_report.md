# Build and review report

日期：2026-10-26

## 产物

- PDF：artifacts/SC-CL_ICLR_draft.pdf
- PDF 页数：9（正文、附录与参考文献合计）；全部为竖版页面（rotation=0）。
- 纸张：US Letter，适配 ICLR 2026 conference style
- 正文展示：方法章节第 3 页嵌入 `artifacts/结构引导与候选级对比优化流程图.svg` 内的 PNG 图像；SVG 原件已逐字节复制到 `main/figures/sccl_structure_guided_source.svg`。该 SVG 是单张 1942×809 PNG 的容器，正文排版尺寸下约 353 ppi；未增加横向页面。另有 NELL prefix-200/full 分层结果表，历史 FB i0 表移至附录。
- 旧占位图：artifacts/SC-CL_MRR_preset.png、artifacts/SC-CL_dataset_delta_preset.png（不再引用）
- PDF SHA-256：880c4818c08ace7a0df7687db4a08b33a5cd5c0738625fe5ee72471633e5fef5

## 编译命令

    cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/11_paper/main
    BSTINPUTS=/mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/11_paper/template/iclr2026: latexmk -g -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build main.tex

## 检查结果

- XeLaTeX/BibTeX/xdvipdfmx：通过。
- 最终 main.log：无硬错误、无未解析 citation/reference、无 overfull hbox；保留少量
  underfull hbox/vbox 和中文字体回退提示。
- 摘要、引言、相关工作、实验、讨论和结论已按“研究问题—方法—可比证据—适用边界”修订；
  对照顶会论文与投稿指南的写法检查见 `artifacts/writing_revision_20261026.md`。
- 本次主线修订明确十选一题目在直接解码与逐候选评分下的不同答案选择规则；
  WN18RR 的单次直接解码结果移入附录，证据与动作见 `artifacts/revision_strategy_20261026_mainline.md`。
- 使用 natbib + ICLR 官方 bst；已目视检查正文第 3 页架构图。排版 PNG 由该 SVG 的唯一嵌入图像无重采样提取；正文图注说明图内损失表达是示意写法，严格公式以正文为准。
- 现有 sampler 单元测试此前已通过：tests/test_counterfactuals.py，5 passed。

## 保留的非阻塞提示

- AR PL UMing CN 没有 small-caps/bold/italic 的完整字形，XeLaTeX 做了字体回退；
  中文正文可读，英文标题/数字使用 TeX Gyre Termes。
- 部分中文长段、附录 schema 和参考文献页有 underfull hbox/vbox，这是当前单栏版心
  下的排版提示，不影响 PDF 输出。

## 科学状态

正文现在按“候选来源问题—受限反事实方法—NELL 运行完整性—prefix-200 配对观察—
生成与排序分歧—证据缺口”组织。NELL23K 的 SC-CL seed=2026 已完成一次训练、train-only
bank、wiring 和 4,943 条测试候选重排；正文明确全量闭集 EM/MRR 91.52%/95.27% 没有全量 B1
候选对照，prefix-200 的 MRR 94.96% 对 93.36% 只是描述性观察。自由生成 EM 为
87.03% 对 B1 的 87.46%，未写成生成收益。FB15K237-2 的历史 i0 移入附录，不被写成
当前 SC-CL 结果。WN18RR 的探索性直接解码 EM 86.89% 只在附录记录；完整重排和公平对照缺失。该 PDF 仍是中期稿。

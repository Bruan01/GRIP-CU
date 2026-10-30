# Build and review report

日期：2026-10-28

- PDF：`artifacts/SC-CL_writing_revision_20261028.pdf`；`artifacts/SC-CL_ICLR_draft.pdf` 同步更新。
- 页数：9；US Letter，竖版。
- PDF SHA-256：`2f358d14d9386dfdad82295fa132b7181b5442669da82409cc2dcb4d0852560d`。
- 编译：XeLaTeX、BibTeX、xdvipdfmx 通过。日志：`artifacts/build_20261028.log`。
- 最终日志无硬错误、未解析引用、缺字或 overfull box；保留中文字体形状回退与部分 underfull 提示。
- 目视检查第 1、3、5、7、8 页；方法图为可编辑 TikZ 矢量图，初始化与离线评分路径独立，联合损失带候选权重。
- 9 个数学公式与修改前一致；20 个标签、18 处交叉引用和 16 个文献键通过检查。
- 本轮修改叙事、结果组织与实际运行设置的表达；没有重新训练或产生新的实验结果。
- 完整修订说明：`artifacts/writing_revision_20261028.md`。
- 修改前源稿与 PDF：`artifacts/before_writing_revision_20261028/`；此前构建报告也在该备份中。

编译命令（从 `main/` 运行）：

```bash
BSTINPUTS=../template/iclr2026: latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

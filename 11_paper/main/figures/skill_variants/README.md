# SC-CL 架构图：三个 skill 的独立方案

三版使用同一份事实约束（`figure_contract.md`），都表达“训练期构建并冻结候选库 → Stage 2 联合训练 → 独立评估”。图中未写入尚未验证的实验收益。论文主图现使用用户提供的 `11_paper/artifacts/结构引导与候选级对比优化流程图.svg`，本目录的三个 skill 版本保留供比较。

| 方案 | 方法与建议用途 | 文件 |
|---|---|
| Nature Figure | `nature-figure` 的可编辑矢量示意图路线；现为备选版本 | `nature_figure_sccl.svg`（原始主文件）、`nature_figure_sccl_paper.svg`（单栏可编辑源）、`nature_figure_sccl_paper.pdf`（排版备选）、相应预览 PNG |
| Framework Figure Studio Pro | v3.2.15c 的语义图约束与内置 `image_gen`；适合视觉方案讨论 | `framework_studio_sccl.png`（模型原图）、`framework_studio_sccl_print_300dpi.png`（排版副本） |
| SciPilot 适配 | 使用其 Nature 样式、导出与 QA 工具制作架构图；适合矢量排版 | `scipilot_adapted_sccl.svg`、`.pdf`、`.png`；`.py` 可重绘，`_grayscale.png` 用于灰度检查 |

## 来源与边界

- 论文依据：`11_paper/main/sections/formulation.tex`、`method.tex`，以及当前候选采样实现。
- Nature 路线进行了 SVG 字号/宽度审计：183 mm，最小字约 6.30 pt，PASS。
- Nature 单栏备选版将图宽改为 139.7 mm，并将标签在最终尺寸下调至约 6.1 pt 以上；其 PDF 为 1.5 版本，字体嵌入。该备选版此前已在编译后的论文页面检查布局。
- Framework 的模型原图为 1860 × 845 px；300 dpi 副本为 2160 × 981 px，系 Lanczos 重采样，细节仍受原图分辨率限制。提交时优先选矢量方案。
- SciPilot 原 skill 明确不覆盖架构图。本版只适配其样式、导出及视觉审计流程，不能称为其原生架构图工作流。SVG 和 600 dpi PNG 检查通过；PDF 的两种 DejaVuSans 字体经 `pdffonts` 确认为嵌入。其 `check_figure.py` 对 CID TrueType 的嵌入检测给出保守 WARN。

## 论文图注候选

**SC-CL training and evaluation pipeline.** A train-only candidate pool excludes the gold relation and relations known for the same entity pair in the training graph. Structural relation profiles and frozen B1 score violations guide a softmax mixture with a uniform floor. Nine distinct candidates per question are fixed before Stage 2. The fixed bank enters candidate InfoNCE alongside the original generation loss. Evaluation reports free-generation EM and candidate-ranking MRR/Hits separately; validation and test signals do not update the bank.

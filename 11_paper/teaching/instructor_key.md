# 教师用答案：候选评估与原论文 EM 的口径混淆

**仅供教师使用，不随学生版 PDF 发放。**学生版是课堂审稿模拟材料，不是可投稿或引用的研究结果。正式研究稿 `../main/main.tex` 保留真实评估口径。

完整学生版为 `SC-CL_full_review_exercise_student.pdf`（8 页），源文件位于 `full_paper_student/`，沿用原论文的章节、架构图与参考文献。此前的 `review_exercise_student.pdf` 是两页短版。发给学生时只提供所选 PDF，不提供本答案或正式稿。

## 核心识别点

学生版在方法中明示 $\hat r=\arg\max_{r\in C_q}z(q,r)$，即对全部候选关系逐项计算平均 token 对数概率，再选最高分。这个数值是 **teacher-forced 候选重排 EM / Hits@1**。GRIP 原论文表 2、表 3 的数值来自 10 选 1 问题上模型生成答案后的 Exact Match。虽然题目都列出十个选项，答案选择规则不同，所以学生版表格中的“官方 EM”及其跨论文增益计算不成立。

## 正确数字

| 数据集与运行 | 候选重排 EM | 生成式 EM | GRIP 原论文 EM |
|---|---:|---:|---:|
| NELL23K / SC-CL / seed 2026 | 91.52% | 87.03% | 87.74% |
| FB15K237-2 / 历史 InfoNCE-i0 | 91.62% | 88.69% | 88.75% |

NELL23K 本地生成评估为 4,943 题，原论文附录报告 4,952 题；FB 本地与原论文均为 10,000 题抽样，但尚无证据证明是同一题目索引。两组数字即使按生成 EM 比较，也需要说明数据和运行协议的差异。FB 历史 i0 不是当前 SC-CL 的跨数据集结果。

## 逐题核对与解释

在保存的同题预测上，NELL23K 有 237 题“生成错、重排对”，其中 235 题的生成答案不在给定候选中；15 题“生成对、重排错”。FB 历史 i0 有 366 题“生成错、重排对”，其中 249 题的生成答案不在候选中；73 题“生成对、重排错”。计数采用 `13_base_method/grip-exp/evaluation/utils.py` 的答案归一化规则。

这些数字表明候选重排的选择约束和完整答案评分可以改变预测，不能把重排得分说成模型按原论文解码规则输出的准确率。NELL 与 FB 的生成提示本身也列出十个选项；差别在于是否让评估程序逐一代入候选打分。

## 建议判分

1. 指出表中“官方 EM”的实现其实是候选重排，而非原论文的生成式 Exact Match。
2. 解释十选一的选项相同，不代表答案选择规则相同；给出上述正确生成 EM 或要求同协议重评。
3. 指出 91.52% 对 87.74%、91.62% 对 88.75% 的增益主张不能成立。
4. 加分：发现 NELL 测试题数不一致、FB i0 非 SC-CL，以及单种子与训练配置差异。

## 证据

- GRIP 原论文：<https://arxiv.org/pdf/2511.07457>，表 2、表 3 及实验设置。
- NELL 结果：`../results_status.md`；生成预测 `../../08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/nell23k_structure_counterfactual_3090_20261021_06_train/listed/predictions_correct.jsonl`；候选重排 `../../08_experiments/Hard_Negative_Contrastive_GRIP/results/candidate_rerank/nell23k_structure_counterfactual_06_full/predictions.jsonl`。
- FB 历史结果：`/mnt/e/GRIP-CU/artifacts/info_nce_fb15k237_qwen7b_20260921/evaluation/i0_generate.jsonl` 与 `i0_candidate_score.jsonl`。

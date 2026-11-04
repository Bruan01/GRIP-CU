# Evidence ledger：SC-CL

| Claim | Evidence location | Status | Safe wording |
|---|---|---|---|
| 2026-11-04 三数据集原冻结搜索完整测试 | `11_paper/artifacts/stage2_three_dataset_completion_20261104.json`；各数据集原始预测、选模记录及独立归档 | observed, seed2026, all three numerical targets achieved | “NELL87.800931%、CoDEx96.576472%、FB89.900000%；各自原先验证选定模型的一次完整测试。单seed本地口径，论文协议对齐仍未全部完成。” |
| 2026-11-04 FB当前Stage2单种子完整测试 | `11_paper/artifacts/fb15k237_2_single_seed_test_20261104.json`；Linux只读硬链接权重与完整预测归档 | observed, seed2026, local protocol; paper audit open | “8990/10000，EM89.90%，数值高于论文参考88.75%约1.15个百分点；匹配B1为81.87%，净增加803题。共同使用已记录的464条train-pair关系QA修复。” |
| 2026-11-04 NELL追加低学习率续训 | `11_paper/artifacts/nell23k_refinement_test_20261104.json`；独立冻结方案与Linux权重归档 | observed negative follow-up; adaptive after prior test exposure | “验证净增24/4950，测试4337/4943（87.740239%），比原冻结模型少3题；本轮未进一步涨点。保留原实验与本次负结果，不将两轮结果隐去后合并。” |
| 2026-11-03 CoDEx-M当前Stage2单种子完整测试 | `11_paper/artifacts/codexm_single_seed_test_20261103.json`；HPO完整预测、冻结选模与Linux只读权重归档 | observed, seed2026, local protocol; paper protocol audit open | “9,958/10,311，EM=96.576472%；数值高于论文参考94.42%约2.156472个百分点。同提示当前B1为94.336146%，净增231题。单种子本地口径，调度及任务协议尚未完全对齐论文。” |
| 2026-11-02 NELL当前Stage2单种子完整测试 | `11_paper/artifacts/nell23k_single_seed_test_20261102.json`；对应HPO `selected_test/predictions.jsonl`、固定选模记录及Linux只读模型归档 | observed, seed2026, local protocol; paper protocol audit open | “4,340/4,943，EM=87.800931%；数值高于论文表3参考87.74%约0.060931个百分点；同提示当前B1为84.584261%。不声称多种子或纯损失隔离收益。” |
| SC-CL sampler 契约 | `08_experiments/Hard_Negative_Contrastive_GRIP/src/hard_negative_grip/counterfactuals.py` | observed implementation | “实现了……” |
| 候选关系只来自 train vocabulary、known-pair 过滤 | sampler、task wiring、NELL bank manifest | observed implementation + NELL run audit | “NELL 运行使用 train-only 过滤……” |
| NELL bank 完成冻结 | `.../results/runs/nell23k_structure_counterfactual_3090_20261021_02_artifacts/structure_counterfactual_bank/metadata.json` | observed | “写入 3,253 行、K=9、rows_failed=0……” |
| NELL wiring 与 hash | `.../structure_counterfactual_bank/wiring.json`、`.../06_train/artifact_manifest.json` | observed | “wiring 和 artifact manifest 通过……” |
| NELL SC-CL Stage 2 完成 | `.../06_train/run_config.json`、`listed/summary.json`、`listed/adapter/` | observed, seed=2026 | “单次 seed=2026 完成……” |
| NELL SC-CL 自由生成 EM | `.../06_train/listed/predictions_correct.jsonl`、`listed/summary.json` | observed | “4,943 条测试 QA，EM=87.032%……” |
| NELL SC-CL full 候选重排 | `results/candidate_rerank/nell23k_structure_counterfactual_06_full/{predictions,summary}.jsonl/json` | observed | “闭集 EM/MRR=91.523%/95.270%……” |
| NELL B1 full 自由生成 | `13_base_method/grip-exp/outputs/grip_inf/nell23k/nell23k_qwen7b_full_seed2026.jsonl` | observed, seed=2026 | “B1 自由生成 EM=87.457%……” |
| NELL B1/SC-CL paired prefix | `candidate_rerank/nell23k_baseline_7b_pilot200` 与 `nell23k_structure_counterfactual_06_full` 前 200 条 | observed, prefix=200 | “同候选前缀上 MRR=93.36% vs 94.96%……” |
| FB B1 full candidate score | `/mnt/e/GRIP-CU/artifacts/info_nce_fb15k237_qwen7b_20260921/evaluation/b1_candidate_score.jsonl` | observed, historical artifact | “FB B1 candidate MRR=88.007%……” |
| FB InfoNCE-i0 full candidate score | `/mnt/e/GRIP-CU/artifacts/info_nce_fb15k237_qwen7b_20260921/evaluation/i0_candidate_score.jsonl` | observed, historical artifact | “i0 candidate MRR=95.412%；不得称为 SC-CL……” |
| FB i0 与当前 SC-CL 契约相同 | historical `grip_run_metadata.json` vs current configs | false / not comparable | 禁止合并成 SC-CL 主结果 |
| 五种子均值/标准差 | 无 2027--2030 formal runs | planned | 只能写为后续协议 |
| Random/Frequency/Profile/Violation/ErrorMemory | 无对应正式结果 | planned | 不能写为对照结论 |
| FB 当前 SC-CL bank | `results/hpo/stage2_three_dataset_20261030/fb15k237_2/assets/k9/{metadata,wiring}.json`；manifest SHA `bbb24960d747f12a6120f541afbc409a3fef4c922e9ae044e316d8fe2952c71c` | observed bank, wiring and selected test | “train-only bank已冻结，wiring保持12,136条生成QA；当前方法单seed完整测试已完成，见本表FB结果行。” |
| WN18RR 单种子探索性直接解码 | `results/hpo/wn18rr_sccl_seed2026/lambda0p25_final_test/summary.json`、`predictions.jsonl`、`search_outcome.json`；本地 B1 `results/runs/wn18rr_sccl_3090_20261023/eval/b1/summary.json` | observed partial | “WN18RR 选定候选损失权重 0.25 后一次全测试 EM=86.89%；本地 B1=32.67%；完整重排和公平对照缺失，原论文五次均值不可作配对收益” |
| CoDEx-M数据协议 | 官方数据、关系文本、全量10-way eval；当前方法结果见本表新增审计行 | observed data; method observed single-seed | CoDEx硬负例不作为关系候选真值；本地单种子生成EM不等同于完整原论文协议复现 |
| ICLR PDF 编译 | `11_paper/artifacts/SC-CL_ICLR_draft.pdf`、build log | observed artifact | “稿件可编译，非科学结果证据” |

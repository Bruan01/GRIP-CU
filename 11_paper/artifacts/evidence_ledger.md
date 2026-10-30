# Evidence ledger：SC-CL

| Claim | Evidence location | Status | Safe wording |
|---|---|---|---|
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
| FB 当前 SC-CL bank | 当前工作区没有可核对的 FB SC-CL run | missing | 必须先运行并通过 wiring |
| WN18RR 单种子探索性直接解码 | `results/hpo/wn18rr_sccl_seed2026/lambda0p25_final_test/summary.json`、`predictions.jsonl`、`search_outcome.json`；本地 B1 `results/runs/wn18rr_sccl_3090_20261023/eval/b1/summary.json` | observed partial | “WN18RR 选定候选损失权重 0.25 后一次全测试 EM=86.89%；本地 B1=32.67%；完整重排和公平对照缺失，原论文五次均值不可作配对收益” |
| CoDEx-M | 官方数据、关系文本、全量 10-way eval 已转换，Stage1/Stage2 pending | planned/partial | 数据资产不等于方法结果或泛化验证；CoDEx 硬负例不作为关系候选真值 |
| ICLR PDF 编译 | `11_paper/artifacts/SC-CL_ICLR_draft.pdf`、build log | observed artifact | “稿件可编译，非科学结果证据” |

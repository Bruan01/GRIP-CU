# FactorGRIP v0.1 — Candidate-Energy Probe

## Decision origin

来自 RecurrentGRIP v1.1.1 的正式诊断：depth matching 只能把 K2 从灾难性退化修复到近似停滞，correct adapter 未超过 no-adapter；输出大量离开候选集合。完整报告：

```text
09_results_analysis/2026-08-30_recurrentgrip_v1_1_1_diagnostic_cross/REPORT.md
```

## Research question

当前低准确率主要来自：

1. graph facts 没有形成可用参数存储；还是
2. graph memory 已影响 hidden state，但自由生成无法选择正确候选 relation？

## Scope

本版本只做 inference-side probe，不重新训练，不修改 Original GRIP，不实现 FactorGRIP experts。

固定使用现有 v1.1.1 train-K1/train-K2 adapter，优先分析 K=1：

- free generation；
- candidate-constrained generation；
- batched candidate sequence-likelihood ranking；
- correct adapter；
- no adapter；
- 若 runner 支持，增加 wrong/shuffled adapter control。

## Required outputs

```text
results/runs/<RUN_ID>/
├── config.json
├── predictions.jsonl
├── candidate_scores.jsonl
├── analysis/
│   ├── decoder_accuracy.csv
│   ├── paired_decoder_effects.csv
│   ├── calibration.json
│   └── summary.json
├── environment.txt
└── run.log
```

每个 question 至少保存：

- `question_id/split/target/candidates`；
- adapter control；
- decoder type；
- 每个候选的长度归一化 log-likelihood；
- predicted candidate；
- correct；
- latency 和 peak memory。

## Fairness rules

- scoring prompt 与 generation prompt 使用同一 question 文本；
- 不读取 target 构造 prompt；
- 候选顺序固定，另做 order permutation robustness；
- 所有候选批量评分，报告额外 FLOPs/latency；
- validation 选择归一化和 temperature，test 配置冻结；
- 不根据当前 64-question test 反复调规则。

## Go / Stop

Go to FactorGRIP v0.2 only if all hold：

1. correct-none test gain ≥ 5 pp；
2. score decoder 相对 free generation test gain ≥ 10 pp；
3. validation/test 方向一致；
4. wrong/shuffled adapter 低于 correct；
5. candidate order permutation 后结论稳定。

否则判定 storage/representation 为主要瓶颈，跳过 decoder 优化，直接实现 equal-rank graph factorization smoke。

## Hardware

- macOS：代码、静态测试、结果分析；
- WSL2 + RTX 3090 24GB：模型加载与 candidate scoring；
- 预计 Qwen2.5-0.5B probe 为 0.5–1.5 GPU 小时；实际 wall time 必须由 runner 外层 monotonic timer 记录。

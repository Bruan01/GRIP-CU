# Source Baseline

## Immediate Parent

- Parent snapshot: `../../RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29`
- Parent experiment ID: `RG-E3-v1.1.1-diagnostic-cross-2026-08-29`
- Fork date: `2026-08-30`
- Copy method: `rsync` of `grip-exp/` and the WSL `configs/` scripts, excluding
  `.venv/`, `__pycache__/`, `*.pyc`, `outputs/`, `model_cache/`, `grip-paper.pdf`.
- Excluded during copy: any `results/`, `logs/`, `.venv/`, `model_cache/`,
  `outputs/`, bytecode, and the 1.2 MB `grip-paper.pdf` (irrelevant to an
  inference-only probe).

## Git Provenance at Fork

- Branch: `wsl/nell23k-smoke-20260829`
- Commit: `2ee0c86`
- Commit subject: `results: add NELL23K diagnostic cross run ...` (analysis +
  FactorGRIP scaffold on top of the diagnostic-cross results commit `8955bf5`).

## Original GRIP Provenance

- Source: `13_base_method/grip-exp`
- Original GRIP mutation: none (inherited unchanged through RecurrentGRIP).

## Version Delta

v0.1 是 RecurrentGRIP 的**诊断派生**，不修改 base method、不重训、不覆盖任何
RecurrentGRIP 版本或已有 run。它复用 v1.1.1 已训练的 `train_k1` / `train_k2`
NELL23K LoRA adapter，只做 inference-side candidate-energy probe：

1. free generation（复刻 v1.1.1 评估路径）；
2. candidate-constrained generation（trie 约束贪心）；
3. batched candidate sequence-likelihood ranking（长度归一化 log-likelihood）；
4. correct / none / wrong-depth adapter control（shuffled 因单图不可用）。

判断当前低准确率主要来自 decoder/retrieval 还是 graph storage 本身，为
FactorGRIP v0.2 的 Go/Stop 提供决策依据。

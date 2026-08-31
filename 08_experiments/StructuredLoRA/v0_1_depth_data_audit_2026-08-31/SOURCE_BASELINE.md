# Source Baseline

## Outer repository

```text
Repository: https://github.com/Bruan01/GRIP-CU.git
Parent commit before this version: c7b9f3d713ec65672d8e1817ddf7c67d593cc24c
Branch at implementation time: wsl/nell23k-smoke-20260829
```

## Original GRIP

```text
Path: 13_base_method/grip-exp/
Commit: 2835b440bfd2c4de36f0380ae19bc1c22e6cb459
Status: clean
```

`13_base_method/grip-exp/` was not modified. This experiment reads only:

```text
data/raw_datasets/nell23k/train.txt
data/raw_datasets/nell23k/valid.txt
data/raw_datasets/nell23k/test.txt
```

Source SHA256 values are frozen in `artifacts/reports/summary.json`.

## Imported diagnostic evidence

Optional baseline-depth analysis reads:

```text
08_experiments/RecurrentGRIP/v1_1_1_diagnostic_cross_2026-08-29/
results/runs/wsl3090_nell23k_diag_cross_20260830_01_nell23k_diagnostic_cross/
predictions.jsonl
```

The imported run is not modified. Its old `true_hop` field is ignored and recomputed support depth is joined from split, endpoint pair, and target relation.

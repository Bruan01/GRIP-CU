# Frozen shared-pool samplers

These manifests are the only negative lists for listed-contrastive
controls. They all read the train-only valid-negative pool. Do not
resample live and do not use the all-split dump under
`20260923_offline_confusion_full`.

## Source

- Scores: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20260923_offline_confusion_train_filter/candidate_scores.jsonl`
- SHA256: `0a4b0a0245e5ccfe6d872ff2aa89b5a2a52aebb7fbaffa7747dce2586e378708`
- Filter splits: `['train']`
- QA count: 3253

## Frozen policy

- Shared pool: `is_valid_negative from train-only dump`
- Random-K / Top-K Hard `k_fixed`: 9
- Coverage-Adaptive `tau`: 0.230804 (median_top9_negative_mass)
- Coverage-Adaptive clamp: `[1, 20]`
- Random seed: 2026
- Soft-Mix: `6` uniform from the full pool + `3` mixture from Top-9, rho=0.33
- Calibrated: lambda_0=1.0, lambda_min=0.25, beta=0.5

Coverage-Adaptive K is **not** K80/K90 over the full 197-way mass. Those
diagnostics stay in the analysis tables (median K80 is far above 9). The
training tau is the median Top-9 `negative_mass`, so the median QA still
uses K=9, concentrated QAs shrink K, and diffuse QAs grow K up to `k_max`.

## Train-only mass / Top-N

- Top-9 mass mean=0.2417 median=0.2308 p25=0.2041 p75=0.2705
- Adaptive K mean=9.315 median=9.0 min=2.0 max=17.0
- Adaptive K < 9: 1168
- Adaptive K = 9: 459
- Adaptive K > 9: 1626

- Calibrated lambda_q mean=0.8791 median=0.8846 min=0.7334 max=0.9287

## Manifests

- `soft_mix`: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261005_shared_pool_truncated_mixture_samplers/soft_mix.jsonl` sha256=`870a90ee5c7d94a961debf4d88a94f7f0e5218d2c55ca689a10b8716b720e1b8`
- `calibrated`: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261005_shared_pool_truncated_mixture_samplers/calibrated.jsonl` sha256=`c1dcb6d50a26b544667cf968527a44fe4262a1bf37b485710d9ace0075b84922`

- Policy: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261005_shared_pool_truncated_mixture_samplers/policy.json`
- Per-QA table: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261005_shared_pool_truncated_mixture_samplers/per_qa.jsonl`
- Global statistics: `/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/Hard_Negative_Contrastive_GRIP/results/runs/20261005_shared_pool_truncated_mixture_samplers/global_statistics.json`

## Selection rules

- Random-K: uniform sample of `k_fixed` from the shared pool.
- Top-K Hard: the `k_fixed` highest `candidate_score` rows in that pool.
- Coverage-Adaptive K: smallest top-K whose cumulative `negative_mass`
  reaches `tau`, clamped to `[k_min, k_max]`.
- Soft-Mix: 6 uniform from the full pool + 3 confusion-mixture draws
  from the Top-N subset, without replacement. This is not Top-3 Hard.
- Calibrated: same negatives as Soft-Mix, with per-QA InfoNCE weight
  `lambda_q = lambda_0 * clip(1 - beta * top9_mass, lambda_min/lambda_0, 1)`.

## Training protocol

- Do not train `64-QA / 10-step listed smoke`.
- Do not train the 3253-QA matchable-only slice.
- CPU-wire the frozen manifests onto the full paper task file before any listed GPU job.
- The next listed run is `soft_mix` on that full task file (~12014 QA, `accum=512`, `epochs=10`, ~230 steps).
- Do not rescore the 3253×198 dump. Do not turn the 64-QA shot into an overnight run.


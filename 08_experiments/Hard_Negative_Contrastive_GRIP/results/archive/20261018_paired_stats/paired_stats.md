# Paired significance and error decomposition

Question-level uncertainty for each trained pair. Bootstrap resamples questions within each split (10,000 draws); McNemar is the two-sided exact test on discordant questions. Neither covers training-seed variance; see the seed table for that.

Pairs not yet available (predictions missing or incomplete): `nell23k_compute_matched`, `nell23k_seed2027`, `fb15k237_seed2027`

## Paired comparison (left vs right; Δ = right − left)

| Pair | Arms | Metric | Split | n | Left | Right | Δ (pp) | 95% CI of Δ | gains / losses | McNemar p |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|
| nell23k | B1 vs Random-K | generate_em | test | 4944 | 84.16 | 88.61 | +4.45 | [+3.52, +5.36] | 394 / 174 | 1e-20 |
| nell23k | B1 vs Random-K | generate_em | validation | 4951 | 83.88 | 89.40 | +5.51 | [+4.61, +6.44] | 422 / 149 | 1e-30 |
| nell23k | B1 vs Random-K | generate_em | all | 9895 | 84.02 | 89.00 | +4.98 | [+4.34, +5.63] | 816 / 323 | 1e-49 |
| nell23k | B1 vs Random-K | closed_set_em | test | 4944 | 87.56 | 92.31 | +4.75 | [+3.94, +5.56] | 338 / 103 | 1e-30 |
| nell23k | B1 vs Random-K | closed_set_em | validation | 4951 | 86.87 | 92.49 | +5.62 | [+4.79, +6.44] | 375 / 97 | 1e-39 |
| nell23k | B1 vs Random-K | closed_set_em | all | 9895 | 87.22 | 92.40 | +5.18 | [+4.59, +5.78] | 713 / 200 | 1e-67 |
| nell23k | B1 vs Random-K | closed_set_mrr | test | 4944 | 0.926 | 0.957 | +0.031 | [+0.026, +0.036] | — | — |
| nell23k | B1 vs Random-K | closed_set_mrr | validation | 4951 | 0.921 | 0.958 | +0.037 | [+0.032, +0.042] | — | — |
| nell23k | B1 vs Random-K | closed_set_mrr | all | 9895 | 0.923 | 0.957 | +0.034 | [+0.031, +0.037] | — | — |
| nell23k | B1 vs Random-K | closed_set_hits@3 | test | 4944 | 97.21 | 99.05 | +1.84 | [+1.42, +2.29] | 107 / 16 | 1e-17 |
| nell23k | B1 vs Random-K | closed_set_hits@3 | validation | 4951 | 96.83 | 99.19 | +2.36 | [+1.90, +2.85] | 132 / 15 | 1e-24 |
| nell23k | B1 vs Random-K | closed_set_hits@3 | all | 9895 | 97.02 | 99.12 | +2.10 | [+1.78, +2.43] | 239 / 31 | 1e-40 |
| fb15k237 | B1 vs Random-K | generate_em | test | 10000 | 74.23 | 82.42 | +8.19 | [+7.31, +9.07] | 1442 / 623 | 1e-74 |
| fb15k237 | B1 vs Random-K | generate_em | validation | 10000 | 74.55 | 83.48 | +8.93 | [+8.06, +9.79] | 1462 / 569 | 1e-89 |
| fb15k237 | B1 vs Random-K | generate_em | all | 20000 | 74.39 | 82.95 | +8.56 | [+7.93, +9.17] | 2904 / 1192 | 1e-162 |
| fb15k237 | B1 vs Random-K | closed_set_em | test | 10000 | 63.40 | 89.58 | +26.18 | [+25.23, +27.13] | 2807 / 189 | 1e-597 |
| fb15k237 | B1 vs Random-K | closed_set_em | validation | 10000 | 63.86 | 88.81 | +24.95 | [+24.02, +25.92] | 2721 / 226 | 1e-542 |
| fb15k237 | B1 vs Random-K | closed_set_em | all | 20000 | 63.63 | 89.20 | +25.56 | [+24.90, +26.24] | 5528 / 415 | 1e-1137 |
| fb15k237 | B1 vs Random-K | closed_set_mrr | test | 10000 | 0.755 | 0.939 | +0.184 | [+0.177, +0.190] | — | — |
| fb15k237 | B1 vs Random-K | closed_set_mrr | validation | 10000 | 0.759 | 0.936 | +0.176 | [+0.170, +0.183] | — | — |
| fb15k237 | B1 vs Random-K | closed_set_mrr | all | 20000 | 0.757 | 0.937 | +0.180 | [+0.176, +0.184] | — | — |
| fb15k237 | B1 vs Random-K | closed_set_hits@3 | test | 10000 | 84.02 | 98.00 | +13.98 | [+13.27, +14.68] | 1434 / 36 | 1e-370 |
| fb15k237 | B1 vs Random-K | closed_set_hits@3 | validation | 10000 | 84.60 | 98.25 | +13.65 | [+12.97, +14.34] | 1395 / 30 | 1e-367 |
| fb15k237 | B1 vs Random-K | closed_set_hits@3 | all | 20000 | 84.31 | 98.12 | +13.81 | [+13.32, +14.30] | 2829 / 66 | 1e-736 |

## Across training seeds (main B1 vs Random-K pairs)

Mean ± sample std over seeds; std is blank with one seed.

| Dataset | Metric | Split | Seeds | B1 | Random-K | Δ (pp) |
|---|---|---|---|---:|---:|---:|
| fb15k237 | generate_em | test | 2026 | 74.23 | 82.42 | 8.19 |
| fb15k237 | generate_em | all | 2026 | 74.39 | 82.95 | 8.56 |
| fb15k237 | closed_set_em | test | 2026 | 63.40 | 89.58 | 26.18 |
| fb15k237 | closed_set_em | all | 2026 | 63.63 | 89.20 | 25.56 |
| nell23k | generate_em | test | 2026 | 84.16 | 88.61 | 4.45 |
| nell23k | generate_em | all | 2026 | 84.02 | 89.00 | 4.98 |
| nell23k | closed_set_em | test | 2026 | 87.56 | 92.31 | 4.75 |
| nell23k | closed_set_em | all | 2026 | 87.22 | 92.40 | 5.18 |

## Free-generation error decomposition (all val+test)

`wrong_in_list`: picked one of the 9 printed distractors. `wrong_off_list_in_vocab`: a real train relation that is not among the 10. `wrong_abstain`: a refusal such as "I don't know". `wrong_oov`: any other string that is not a train relation. Membership uses the EM normalization.

| Pair | Arm | n | correct | wrong_in_list | wrong_off_list_in_vocab | wrong_abstain | wrong_oov |
|---|---|---:|---:|---:|---:|---:|---:|
| nell23k | B1 | 9895 | 8314 (84.0%) | 1074 (10.9%) | 421 (4.3%) | 5 (0.1%) | 81 (0.8%) |
| nell23k | Random-K | 9895 | 8807 (89.0%) | 615 (6.2%) | 270 (2.7%) | 1 (0.0%) | 202 (2.0%) |
| fb15k237 | B1 | 20000 | 14878 (74.4%) | 2642 (13.2%) | 376 (1.9%) | 1506 (7.5%) | 598 (3.0%) |
| fb15k237 | Random-K | 20000 | 16590 (83.0%) | 1132 (5.7%) | 404 (2.0%) | 837 (4.2%) | 1037 (5.2%) |

## Transitions left → right (all val+test)

**nell23k** (rows B1, columns Random-K)

| left \ right | correct | wrong_in_list | wrong_off_list_in_vocab | wrong_abstain | wrong_oov |
|---|---:|---:|---:|---:|---:|
| correct | 7991 | 146 | 109 | 0 | 68 |
| wrong_in_list | 526 | 419 | 50 | 0 | 79 |
| wrong_off_list_in_vocab | 236 | 46 | 93 | 1 | 45 |
| wrong_abstain | 3 | 0 | 2 | 0 | 0 |
| wrong_oov | 51 | 4 | 16 | 0 | 10 |

**fb15k237** (rows B1, columns Random-K)

| left \ right | correct | wrong_in_list | wrong_off_list_in_vocab | wrong_abstain | wrong_oov |
|---|---:|---:|---:|---:|---:|
| correct | 13686 | 292 | 194 | 206 | 500 |
| wrong_in_list | 1535 | 666 | 71 | 125 | 245 |
| wrong_off_list_in_vocab | 173 | 33 | 63 | 19 | 88 |
| wrong_abstain | 785 | 118 | 44 | 472 | 87 |
| wrong_oov | 411 | 23 | 32 | 15 | 117 |


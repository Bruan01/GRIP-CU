# v0.1.2 Controlled Protocol Results Analysis

- Date: 2026-09-01
- Environment: WSL2 + conda `guardenv` + RTX 3090 24GB
- Seed: 42
- Test size: 156
- Protocols: `direct_answer_only` vs `oracle_two_stage`

## Main result

| protocol | selected checkpoint | graph-free validation | graph-free test | correct/test |
|---|---:|---:|---:|---:|
| direct answer-only | stage2_epoch5 | 0.3224 | 0.2692 | 42/156 |
| oracle two-stage | stage2_epoch5 | 0.3026 | 0.3269 | 51/156 |

On this single seed, oracle two-stage is **+5.77 percentage points** on held-out test (51 vs 42 correct). This is promising but inconclusive: direct has the better validation score, both sets are small, and only one seed exists. Approximate 95% binomial intervals are broad: direct 20.6–34.4%, oracle 25.8–40.4%.

## Full curve

| checkpoint | direct validation | direct test | oracle validation | oracle test |
|---|---:|---:|---:|---:|
| direct warm / oracle stage1 | 0.2566 / 0.0000 | 0.2821 / 0.0000 | 0.0000 / 1.0000 | 0.0000 / 1.0000 |
| epoch 1 | 0.1776 | 0.2115 | 0.1579 | 0.1731 |
| epoch 2 | 0.2500 | 0.2564 | 0.1974 | 0.2628 |
| epoch 3 | 0.2566 | 0.2949 | 0.2632 | 0.2628 |
| epoch 4 | 0.2632 | 0.3205 | 0.2829 | 0.3141 |
| epoch 5 | **0.3224** | 0.2692 | 0.3026 | 0.3269 |
| epoch 6 | 0.2961 | 0.2885 | 0.2632 | 0.2885 |
| epoch 7 | 0.2961 | **0.3462** | 0.2829 | **0.3526** |
| epoch 8 | 0.2895 | 0.2372 | 0.2500 | 0.2756 |

Epoch 7 is displayed only to show instability. It cannot be the official result because it was identified using test labels. The official selected result uses validation only and is epoch 5 for both protocols.

## What this verifies

1. **Stage 1 is not necessary.** Direct answer-only already reaches 28.2% test after its matched first budget segment and reaches 34.6% at the best observed test checkpoint.
2. **Stage 1 may help, but the evidence is weak.** At the validation-selected checkpoint, oracle is 32.7% test versus direct 26.9%, a +5.8 point single-seed difference.
3. **The gain is not stable across checkpoints.** Oracle is lower at epochs 1 and 3, tied at epoch 6, and higher at epochs 2, 4, 5, 7, and 8. The curves do not establish a robust causal benefit.
4. **The shortcut diagnosis remains.** Oracle Stage 1 is 100% with visible gold evidence but 0% graph-free. Direct's first segment is already 28.2% graph-free. Oracle spends its first budget learning evidence-conditioned behavior and must recover during Stage 2.
5. **This is not a code failure.** Loss decreases and accuracy changes in a plausible way; the main issue is protocol/optimization variance and weak generalization from a small single-seed evaluation.

## Defensible conclusion

> Under one controlled seed, oracle two-stage gives a higher validation-selected held-out test score than direct answer-only (32.69% vs 26.92%), but the advantage is not established because validation ranks direct higher, the curves are non-monotonic, and only one seed has been run.

The result is enough to justify repeating the comparison, but not enough to implement a learned prioritizer.

## Next action

Run seeds 43 and 44 for both protocols with identical configs. Aggregate validation-selected test accuracy, mean/std, depth breakdown, and paired per-example differences. Then add anti-copy diagnostics. If oracle does not consistently beat direct, simplify to direct answer-only or redesign Stage 1 instead of adding a learned scorer.

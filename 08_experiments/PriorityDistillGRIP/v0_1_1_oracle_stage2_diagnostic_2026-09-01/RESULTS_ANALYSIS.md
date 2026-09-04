# v0.1.1 Oracle Stage-2 Diagnostic Results

- **Run ID**: `wsl3090_oracle_stage2_diagnostic_20260901_01`
- **Date**: 2026-09-01 (Asia/Shanghai)
- **Environment**: WSL2 + conda `guardenv` + RTX 3090 24GB
- **Model**: local `Qwen/Qwen2.5-0.5B-Instruct`
- **Decision**: diagnostic evidence for Stage-2 undertraining, but no clean mechanism claim yet

## Curve

| checkpoint | graph-free test | graph-free validation | oracle-evidence test | oracle-evidence validation |
|---|---:|---:|---:|---:|
| initial | 0.0000 | 0.0000 | 0.2244 | 0.1513 |
| stage1_end | 0.0000 | 0.0000 | **1.0000** | **1.0000** |
| stage2_epoch1 | 0.1731 | 0.1513 | 0.5192 | 0.5066 |
| stage2_epoch2 | 0.2436 | 0.2171 | 0.5192 | 0.4934 |
| stage2_epoch3 | 0.2628 | 0.2763 | 0.4615 | 0.4803 |
| stage2_epoch4 | 0.2756 | 0.2829 | 0.4295 | 0.4276 |
| stage2_epoch5 | **0.3590** | **0.3224** | 0.5641 | 0.5724 |
| stage2_epoch6 | 0.3141 | 0.2632 | 0.5128 | 0.4342 |
| stage2_epoch7 | 0.3462 | 0.3026 | 0.3590 | 0.3092 |
| stage2_epoch8 | 0.3013 | 0.2895 | 0.4167 | 0.4408 |

The original v0.1 answer-only test accuracy was `0.3141`; therefore Stage-2 epoch 5 is `+0.0449` absolute on this single seed. This is a checkpoint selection result, not a final multi-seed claim.

## Findings

### 1. This was not a simple failure to optimize

At the end of Stage 1, oracle-evidence accuracy reached `100%` on both validation and test. Stage-1 loss was `0.1301`, with zero prompt truncation. The model learned the evidence-conditioned task extremely well.

At the same checkpoint, graph-free accuracy was `0%` on both splits. This directly demonstrates that Stage-1 fitting did not produce graph-free behavior.

### 2. Stage 2 training does recover graph-free behavior

Graph-free test accuracy improved from `0%` at Stage-1 end to:

```text
17.31% → 24.36% → 26.28% → 27.56% → 35.90%
```

through Stage-2 epoch 5. Validation followed the same broad trend and peaked at `32.24%` at epoch 5. This is strong evidence that the original v0.1 Stage 2 budget was too short to fully adapt away from the evidence-conditioned input distribution.

### 3. The recovery is not monotonic and late epochs degrade

After epoch 5, graph-free test accuracy fell to `31.41%` at epoch 6, `34.62%` at epoch 7, and `30.13%` at epoch 8. Oracle-evidence accuracy also degraded from `100%` at Stage-1 end to `41.67%` at epoch 8.

Therefore, simply training longer is not sufficient. The current procedure shows a useful intermediate checkpoint around epoch 5, followed by instability/forgetting. A later protocol needs validation-based checkpoint selection, a smaller Stage-2 learning rate, and/or mixed replay of Stage-1 evidence examples.

### 4. The shortcut hypothesis is supported

The combination of:

- Stage-1 oracle-evidence `1.0000`;
- Stage-1 graph-free `0.0000`;
- loss of oracle-evidence accuracy during answer-only Stage 2;
- gradual graph-free recovery during the same Stage 2;

is consistent with the model first learning to use/copy visible path evidence, then partially learning the graph-free answer mapping during Stage 2. It does not prove that every gain is shortcut-driven, but it rules out interpreting the low Stage-1 loss as graph-free internalization.

## Limitations

- One seed only;
- epoch 5 was observed after looking at the curve, so it cannot be treated as an unbiased test estimate;
- graph-free and oracle-evidence prompts are different input conditions by design;
- Stage 2 uses the answer-only objective, so this diagnostic does not test anti-copy data formats;
- no comparison control was rerun with the same 8-epoch schedule.

## Next decision

Do not implement a learned prioritizer yet. The next experiment should be a **controlled protocol revision**:

1. split validation from test for checkpoint selection;
2. use 4/8/12 Stage-2 epochs as pre-registered candidates or select only on validation;
3. lower Stage-2 LR from `1e-3` and compare constant vs cosine schedules;
4. add a small Stage-1 evidence replay term during Stage 2 to reduce catastrophic forgetting;
5. add an anti-copy condition that hides/replaces the terminal path node;
6. rerun at least answer-only, More-QA, and oracle under the same schedule before any learned scorer.

The current result changes the diagnosis from “oracle did not work” to:

> **The oracle path objective fits, but it first creates evidence-dependent copying. A longer Stage 2 can recover graph-free accuracy, with the best observed single-seed checkpoint at epoch 5, but the recovery is unstable and not yet a clean causal result.**

# ErrorMemory-9 sampling design

Date: 2026-10-19
Status: implemented as a frozen train-only sampler; no GPU run has been started.

## Design decision

Use the original GRIP generation-only B1 as the teacher. For each train
relation question, build a legal relation pool from the train vocabulary,
remove the gold and known train facts for the same entity pair, and sample
`K=9` without replacement from:

```text
u(q,r) = alpha * E(gold(q), r) + beta * V_B1(q,r)
p(r|q) = rho / |P_q| + (1-rho) * softmax(u(q,r) / tau)
```

`E` is gold-conditioned B1 error frequency, `V_B1` is the B1 pairwise score
violation, and the uniform floor prevents collapse to a few frequent
confusions. Defaults are `rho=1/3`, `alpha=1`, `beta=1`, `tau=0.25`.

## Why this is a useful test

Uniform Random-9 tests whether InfoNCE alone helps. ErrorMemory-9 tests
whether model-specific boundary evidence adds value at the same `K`, prompt,
generation objective, Stage 2 epochs, optimizer, seed and candidate scorer.
The sampler is frozen before training, so the experiment does not introduce
an online feedback loop.

## Risk controls

- only `train.txt` may contribute error memory, scores and known-fact filters;
- validation/test facts are not used for candidate construction;
- a candidate absent from train is not guaranteed false in an open-world KG;
- exact question IDs and baseline adapter hashes are required across rollout,
  score, freeze and training;
- `wiring.json` must show complete task coverage and `K=9` before training.

## Excluded from the first experiment

No adaptive `K`, no structural/path sampler, no unlikelihood term, no online
resampling, and no extra generation-side penalty. Those would change the
story or the objective and belong to separate ablations.

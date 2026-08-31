# StructuredLoRA Project Memory

Updated: 2026-08-31

## Current decision

StructuredLoRA is accepted only as an **oracle-prefix candidate**. The next critical question is whether ordered cumulative rank groups beat a parameter-matched monolithic LoRA under perfect routing.

## Do not regress to the weak idea

The publishable center is not:

```text
split LoRA rank + experts + router + orthogonal initialization
```

Those components have close prior art. The candidate contribution is:

```text
ordered cumulative depth subspaces
+ ordinal routing P(depth >= g)
+ depth-local residual credit assignment
+ causal group knockout evidence
```

## Frozen terminology

- `query depth`: explicit number of reasoning steps demanded by a controlled path QA;
- `support depth`: shortest train-graph path supporting a NELL23K relation-prediction pair;
- NELL23K support depth is a structural proxy, not ground-truth query hop;
- `>4_or_unreachable` is a censored bucket, not proof of global disconnection.

## Fairness requirements for v0.2+

- same backbone;
- same train tasks and token budget;
- same LoRA target modules;
- same total rank and adapter parameter budget;
- same optimizer, epochs, seed and decoder;
- include monolithic + depth auxiliary head to isolate auxiliary-supervision gains;
- include flat MoE/rank-expert and static all-on split baselines;
- do not compare only S-LoRA rank 12 against Original GRIP rank 4.

## Stop rule

Stop StructuredLoRA if oracle ordered prefix does not beat equal-rank monolithic LoRA on exact-hop validation/test with the same training data. A learned router cannot rescue a structure that loses under perfect routing.

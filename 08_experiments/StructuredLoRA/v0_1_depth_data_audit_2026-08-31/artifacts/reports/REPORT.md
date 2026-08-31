# StructuredLoRA v0.1 Depth Data Audit Report

- Dataset: `NELL23K`
- Train/valid/test: `24321/4951/4944`
- Train graph nodes/edges/relations: `20789/24321/198`
- Decision: **GO_ORACLE_PREFIX**

## What was measured

1. Train triples use leave-one-edge-out support depth.
2. Validation and test triples use shortest support paths in the train graph.
3. Directed and undirected depths are both exported; undirected is the primary NELL23K structural proxy.
4. Exact-hop tasks require a simple directed path, no shorter directed path, and a unique answer for the relation chain.
5. Existing diagnostic predictions are joined by split, endpoint pair, and target relation; their previous `true_hop` field is not trusted.

## Decision gates

| Gate | Result |
|---|---|
| `exact_hop_generation_complete` | PASS |
| `at_least_two_nontrivial_test_buckets` | PASS |
| `depth_not_relation_identity` | PASS |
| `existing_baseline_has_depth_variation` | PASS |

## Key measurements

- Test nontrivial support buckets with at least 100 examples: `2, 3, 4`
- Test relation-depth normalized mutual information: `0.159158`
- Largest observed baseline accuracy spread across usable depth buckets: `0.14912280701754385`
- Exact-hop tasks: `1024`
- Prediction join rate: `1.0`

## Interpretation

Proceed only to an oracle-prefix, equal-rank pilot. Learned routing remains blocked until the oracle structure beats monolithic LoRA.

The NELL23K label is a **support-depth proxy**, not a controlled query-hop label. It can support the performance table and stratified analysis, but exact-hop data is still required for causal depth-specialization claims.

The imported diagnostic run contains only 32 validation and 64 test questions per condition. Its depth-wise accuracy variation is a weak feasibility signal, not paper-level statistical evidence.

A bounded `>4_or_unreachable` bucket means no path was found within four steps; it must not be described as globally unreachable.

## Next experiment

Run `v0_2_oracle_prefix_smoke` with identical data, token budget, target modules, and total LoRA rank for every baseline:

1. equal-rank monolithic LoRA;
2. static split with all groups active;
3. flat routed experts;
4. ordered oracle prefix;
5. ordered oracle prefix plus depth-local credit assignment;
6. permuted depth labels and random group order controls.

Do not implement the learned router unless oracle prefix beats equal-rank monolithic LoRA.

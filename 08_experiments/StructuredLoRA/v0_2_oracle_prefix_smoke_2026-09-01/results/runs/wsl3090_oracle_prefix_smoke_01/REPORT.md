# StructuredLoRA v0.2 Suite Report

Decision: **PRELIMINARY_STOP**

## Test metrics

| Method | Seeds | Accuracy | Macro-hop | Worst-hop | d1 | d2 | d3 | d4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_oracle | 1 | 0.2949 | 0.2949 | 0.1538 | 0.1538 | 0.1795 | 0.4103 | 0.4359 |
| monolithic | 1 | 0.3462 | 0.3462 | 0.1538 | 0.1538 | 0.2821 | 0.5128 | 0.4359 |
| non_nested_random_masks | 1 | 0.2949 | 0.2949 | 0.1026 | 0.1026 | 0.1538 | 0.4359 | 0.4872 |
| ordered_prefix | 1 | 0.3205 | 0.3205 | 0.1538 | 0.1538 | 0.2051 | 0.4615 | 0.4615 |
| ordered_prefix_local_credit | 1 | 0.2949 | 0.2949 | 0.1282 | 0.1282 | 0.1795 | 0.4103 | 0.4615 |
| permuted_depth_prefix | 1 | 0.3462 | 0.3462 | 0.1795 | 0.1795 | 0.2308 | 0.5128 | 0.4615 |
| random_group_order | 1 | 0.3205 | 0.3205 | 0.1538 | 0.1538 | 0.2051 | 0.4359 | 0.4872 |
| static_split | 1 | 0.3782 | 0.3782 | 0.1795 | 0.1795 | 0.3077 | 0.5385 | 0.4872 |

## Registered gate

- `ordered_over_monolithic`: value `-0.025641`, threshold `0.000000`, pass `False`
- `ordered_over_flat`: value `0.025641`, threshold `0.000000`, pass `True`
- `deep_3_4_improvement`: value `-0.012821`, threshold `0.030000`, pass `False`
- `one_hop_not_degraded`: value `0.000000`, threshold `-0.010000`, pass `True`
- `ordered_over_permuted`: value `-0.025641`, threshold `0.000000`, pass `False`
- `ordered_over_non_nested`: value `0.025641`, threshold `0.000000`, pass `True`

## Control interpretation

A fixed permutation of group identities preserves the nested prefix function class. It is a symmetry control, not a valid expected-degradation control; the non-nested mask is decisive.

## Stop rule

Only `GO_LEARNED_ROUTER` permits implementation of a learned ordinal router. `PRELIMINARY_GO` requires the remaining registered seeds first. Any stop decision is recorded without adding rank.

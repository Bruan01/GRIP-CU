# PriorityDistill-GRIP v0.1 Suite Report

Decision: **PRELIMINARY_STOP**

| Method | Seeds | Accuracy | Deep 3/4 | d1 | d2 | d3 | d4 | Stage-1 tokens | Stage-1 steps | Truncated examples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_only | 1 | 0.3141 | 0.4872 | 0.1538 | 0.1282 | 0.5385 | 0.4359 | 0 | 0.0 | 0.00% |
| more_qa_equal_token | 1 | 0.3526 | 0.5128 | 0.1795 | 0.2051 | 0.6154 | 0.4103 | 199366 | 93.0 | 0.00% |
| random_path_equal_token | 1 | 0.3397 | 0.4872 | 0.1538 | 0.2308 | 0.4615 | 0.5128 | 199337 | 89.0 | 0.00% |
| all_paths_equal_token | 1 | 0.3397 | 0.4744 | 0.1795 | 0.2308 | 0.4872 | 0.4615 | 199315 | 83.0 | 0.00% |
| oracle_priority_equal_token | 1 | 0.3141 | 0.4487 | 0.1538 | 0.2051 | 0.4615 | 0.4359 | 199321 | 89.0 | 0.00% |

## Registered gate

- `oracle_over_answer_only`: value `0.000000`, comparison `greater_equal`, threshold `0.020000`, pass `False`
- `oracle_over_more_qa`: value `-0.038462`, comparison `greater_equal`, threshold `0.010000`, pass `False`
- `oracle_over_random_path`: value `-0.025641`, comparison `greater`, threshold `0.000000`, pass `False`
- `oracle_over_all_paths`: value `-0.025641`, comparison `greater`, threshold `0.000000`, pass `False`
- `deep_3_4_improvement`: value `-0.038462`, comparison `greater`, threshold `0.000000`, pass `False`
- `stage1_token_budget_match`: value `0.000256`, comparison `less_equal`, threshold `0.010000`, pass `True`
- `stage1_no_prompt_truncation`: value `0.000000`, comparison `less_equal`, threshold `0.000000`, pass `True`

## Interpretation boundary

Only `GO_LEARNED_PRIORITIZER` authorizes v0.2 learned path scoring. A one-seed result is preliminary. If token budgets diverge materially, the suite remains a systems result rather than a mechanism claim.

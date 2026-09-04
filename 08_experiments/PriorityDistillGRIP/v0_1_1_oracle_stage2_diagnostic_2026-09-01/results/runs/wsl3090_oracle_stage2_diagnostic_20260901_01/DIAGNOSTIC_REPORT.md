# Oracle Stage-2 diagnostic curve

`graph_free` is the actual intended evaluation. `oracle_evidence` is diagnostic-only and exposes the row's gold path.

| checkpoint | condition/split | accuracy | deep 3/4 |
|---|---|---:|---:|
| initial | graph_free_test | 0.0000 | 0.0000 |
| initial | graph_free_validation | 0.0000 | 0.0000 |
| initial | oracle_evidence_test | 0.2244 | 0.0769 |
| initial | oracle_evidence_validation | 0.1513 | 0.0526 |
| stage1_end | graph_free_test | 0.0000 | 0.0000 |
| stage1_end | graph_free_validation | 0.0000 | 0.0000 |
| stage1_end | oracle_evidence_test | 1.0000 | 1.0000 |
| stage1_end | oracle_evidence_validation | 1.0000 | 1.0000 |
| stage2_epoch1 | graph_free_test | 0.1731 | 0.2692 |
| stage2_epoch1 | graph_free_validation | 0.1513 | 0.1579 |
| stage2_epoch1 | oracle_evidence_test | 0.5192 | 0.5128 |
| stage2_epoch1 | oracle_evidence_validation | 0.5066 | 0.4605 |
| stage2_epoch2 | graph_free_test | 0.2436 | 0.3718 |
| stage2_epoch2 | graph_free_validation | 0.2171 | 0.2500 |
| stage2_epoch2 | oracle_evidence_test | 0.5192 | 0.6154 |
| stage2_epoch2 | oracle_evidence_validation | 0.4934 | 0.5263 |
| stage2_epoch3 | graph_free_test | 0.2628 | 0.3333 |
| stage2_epoch3 | graph_free_validation | 0.2763 | 0.3816 |
| stage2_epoch3 | oracle_evidence_test | 0.4615 | 0.5128 |
| stage2_epoch3 | oracle_evidence_validation | 0.4803 | 0.4474 |
| stage2_epoch4 | graph_free_test | 0.2756 | 0.3846 |
| stage2_epoch4 | graph_free_validation | 0.2829 | 0.3553 |
| stage2_epoch4 | oracle_evidence_test | 0.4295 | 0.5000 |
| stage2_epoch4 | oracle_evidence_validation | 0.4276 | 0.4737 |
| stage2_epoch5 | graph_free_test | 0.3590 | 0.4872 |
| stage2_epoch5 | graph_free_validation | 0.3224 | 0.4474 |
| stage2_epoch5 | oracle_evidence_test | 0.5641 | 0.6667 |
| stage2_epoch5 | oracle_evidence_validation | 0.5724 | 0.6579 |
| stage2_epoch6 | graph_free_test | 0.3141 | 0.4487 |
| stage2_epoch6 | graph_free_validation | 0.2632 | 0.3421 |
| stage2_epoch6 | oracle_evidence_test | 0.5128 | 0.6026 |
| stage2_epoch6 | oracle_evidence_validation | 0.4342 | 0.4342 |
| stage2_epoch7 | graph_free_test | 0.3462 | 0.4872 |
| stage2_epoch7 | graph_free_validation | 0.3026 | 0.3947 |
| stage2_epoch7 | oracle_evidence_test | 0.3590 | 0.4615 |
| stage2_epoch7 | oracle_evidence_validation | 0.3092 | 0.3684 |
| stage2_epoch8 | graph_free_test | 0.3013 | 0.4359 |
| stage2_epoch8 | graph_free_validation | 0.2895 | 0.3816 |
| stage2_epoch8 | oracle_evidence_test | 0.4167 | 0.5000 |
| stage2_epoch8 | oracle_evidence_validation | 0.4408 | 0.4474 |

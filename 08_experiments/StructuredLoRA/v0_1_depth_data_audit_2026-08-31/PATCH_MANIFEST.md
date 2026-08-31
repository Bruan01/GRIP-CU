# Patch Manifest

## Scope

This version adds a standalone CPU-only data audit under `08_experiments/StructuredLoRA/`. It does not modify Original GRIP or old RecurrentGRIP/FactorGRIP snapshots.

## Code

```text
scripts/run_depth_data_audit.py
structured_lora_audit/io_utils.py
structured_lora_audit/graph.py
structured_lora_audit/depth.py
structured_lora_audit/exact_hop.py
structured_lora_audit/predictions.py
structured_lora_audit/report.py
```

## Tests

```text
tests/test_graph.py
tests/test_exact_hop.py
tests/test_predictions.py
```

The tests cover leave-one-edge-out removal, multi-relation pairs, directed versus undirected distance, censored buckets, exact-hop provenance, deterministic generation, and diagnostic prediction joins.

## Configuration

```text
configs/depth_audit_nell23k.json
configs/run_depth_audit.sh
configs/check_mac_static.sh
```

## Generated and committed evidence

```text
artifacts/depth_labels/
artifacts/exact_hop/
artifacts/reports/
```

The compressed full support-depth label file uses deterministic gzip metadata (`mtime=0`), so identical source data and configuration reproduce the same SHA256.

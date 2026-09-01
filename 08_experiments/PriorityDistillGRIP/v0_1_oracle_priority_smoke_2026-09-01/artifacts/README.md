# Artifacts

- `setup_audit.json`: static design invariants and source-data SHA256.
- `supervision/candidate_pools.jsonl`: deterministic train-only candidate pools.
- `supervision/<method>.jsonl`: five registered supervision variants.
- `supervision/audit.json`: artifact sizes, hashes, depth counts, and leakage boundary.

Regenerate deterministically with:

```bash
python3 scripts/build_supervision.py
python3 scripts/validate_setup.py
```

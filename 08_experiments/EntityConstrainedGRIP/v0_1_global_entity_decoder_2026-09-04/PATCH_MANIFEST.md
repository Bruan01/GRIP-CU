# Patch Manifest — 2026-09-04 Validation-First Revision

## Scope

```text
08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/
README.md
TODO_ENTITY_CONSTRAINED_GRIP.md
```

## Revised protocol

- primary checkpoints：direct seed43/44；
- reference：More-QA seed42；joint seed43/44 secondary-only；
- initial split：validation only；
- initial decoder：D0 frozen artifact reuse + D1 train-KG global trie；
- D0 artifacts：SHA256 + exact task_id + answer provenance validation；
- mechanism probe：D0→D1 invalid/valid-wrong/correct transitions；
- gate：mean canonical `≥ +2 pp`，per-checkpoint raw/canonical nonnegative，mean novel-composition `≥ -1 pp`；
- D2：仅 `PRELIMINARY_GO_D2` 后运行；
- D3：explicit diagnostic-only mode；
- Phase-A runner：不打开 test。

## Files added or materially revised

- `entity_decoder/artifacts.py`
- `entity_decoder/metrics.py`
- `entity_decoder/gate.py`
- `entity_decoder/config.py`
- `scripts/run_decoder_smoke.py`
- `scripts/summarize_smoke.py`
- `scripts/validate_setup.py`
- `scripts/runtime_self_test.py`
- `configs/phase_a_decoder_smoke.json`
- `configs/run_wsl_smoke.sh`
- `configs/check_wsl_runtime.sh`
- `tests/test_artifacts.py`
- `tests/test_metrics.py`
- `tests/test_gate.py`
- `tests/test_config.py`
- protocol documentation in this directory.

## Immutable/excluded

- no files changed under `13_base_method/grip-exp/`;
- no existing run directory overwritten;
- unrelated untracked files untouched;
- checkpoint `.pt` files remain ignored and resident on the original WSL machine;
- no commit or push performed in this revision step.

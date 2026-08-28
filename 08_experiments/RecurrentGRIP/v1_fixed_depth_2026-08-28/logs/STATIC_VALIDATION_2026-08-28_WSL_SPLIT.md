# Static Validation — macOS / WSL2 Split — 2026-08-28

## Validation scope

This validation freezes the platform boundary for `RG-E1-v1-fixed-depth-2026-08-28`:

- macOS is the editing, versioning, static-check, and result-review environment;
- Windows WSL2 with RTX 3090 24GB is the ML integration, smoke, and pilot environment;
- macOS CPU/MPS output is not treated as a paper result;
- Original GRIP remains an immutable baseline source.

## Original GRIP isolation

```text
repository: 13_base_method/grip-exp
HEAD:       2835b440bfd2c4de36f0380ae19bc1c22e6cb459
status:     clean
```

Result: PASS. RecurrentGRIP code and platform scripts exist only in the independent version snapshot under `08_experiments/RecurrentGRIP/`.

## macOS static validation

Command:

```bash
cd /Users/mac/CODE/GRIP/ai_research_workflow/08_experiments/RecurrentGRIP/v1_fixed_depth_2026-08-28
bash configs/check_mac_static.sh
```

Observed output:

```text
AST syntax check: PASS (83 Python files)
macOS static checks: PASS
```

The static checker parses Python with `ast`, validates every version-local Bash script with `bash -n`, and validates `configs/pilot_qwen05b.json`. It does not install Torch or create project-local Python bytecode.

## Artifact hygiene

Removed development-time `__pycache__/`, `*.pyc`, and `*.pyo` artifacts from the snapshot. A post-check scan returned no matches.

New-version copying also excludes:

```text
.venv/
model_cache/
outputs/
results/*
logs/*
__pycache__/
*.pyc
*.pyo
```

Result: PASS. macOS and WSL environments, model caches, previous runs, and Python bytecode are not inherited by later method versions.

## Runtime fixes included before WSL execution

1. **Evaluation device placement** — the base model reconstructed for adapter evaluation is explicitly moved to the selected CUDA device after loading the adapter.
2. **CUDA guard** — WSL smoke/pilot launchers use `--evaluation_device cuda --require_cuda true`, so a formal run does not silently fall back to CPU.
3. **Immutable run directories** — every smoke/pilot creates `results/runs/<RUN_ID>_*`; an existing path causes an error instead of overwriting results.
4. **Hard wall-time control** — Python-level limits are backed by WSL GNU `timeout`, including a TERM-to-KILL grace period.
5. **Environment capture** — every run records `nvidia-smi`, Python, PyTorch/CUDA verification, and frozen dependencies in `environment.txt`.

## WSL2 RTX 3090 checks still pending

The following items require the execution machine and are not claimed by this macOS validation:

- CUDA PyTorch / Transformers / PEFT installation;
- RTX 3090 name, approximately 24 GiB VRAM, CUDA availability, and BF16 verification;
- full unit-test suite;
- tiny PEFT adapter-scope and adapter reload/device-placement tests;
- Qwen2.5-0.5B one-graph smoke;
- two-hour CLEGR pilot.

## Required WSL execution order

```bash
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
CLEGR_INPUT=/path/to/recurrent_station_shortest.json \
  bash configs/run_qwen05b_smoke_wsl.sh
```

Only after the smoke run passes should the pilot be launched:

```bash
CLEGR_INPUT=/path/to/recurrent_station_shortest.json \
  bash configs/run_qwen05b_pilot.sh
```

## Validation conclusion

```text
macOS static validation:       PASS
Original GRIP isolation:       PASS
snapshot artifact hygiene:     PASS
WSL2 ML integration:           PENDING
RTX 3090 smoke/pilot:          PENDING
```

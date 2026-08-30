# WSL2 + RTX 3090 Environment — FactorGRIP v0.1 Candidate-Energy Probe

## Scope and isolation

This document describes the inference-only NELL23K probe in
`08_experiments/FactorGRIP/v0_1_candidate_energy_probe_2026-08-30/`.
It does **not** modify `13_base_method/grip-exp/`, retrain a model, run FactorGRIP
experts, or overwrite RecurrentGRIP runs. The probe uses the existing
RecurrentGRIP v1.1.1 `train_k1` and `train_k2` adapters as `correct` and
`wrong_depth` controls, plus a `none` control. A shuffled-adapter control is
reported as unavailable because the prepared input contains one NELL23K graph.

The WSL run must use the existing conda environment **`guardenv`**. The scripts
never create or use a project `.venv`.

## Platform split

- macOS or CPU-only checkout: source inspection, JSON/Bash/AST checks, pure
  Python unit tests, and analysis fixtures;
- WSL2 + RTX 3090 24GB: Qwen2.5-0.5B loading, adapter inference, free and
  candidate-constrained decoding, and batched candidate scoring;
- do not copy `.venv`, conda environments, model caches, or generated run
  directories between platforms.

## Environment setup and verification

Activate or expose conda, then verify the named environment:

```bash
conda activate guardenv
python --version
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The repository scripts can also invoke the environment without activation:

```bash
CONDA_ENV=guardenv bash configs/setup_wsl3090.sh
CONDA_ENV=guardenv bash configs/test_wsl3090.sh
```

`test_wsl3090.sh` intentionally fails when CUDA is unavailable; it does not
silently fall back to CPU for the GPU experiment. On the development runner
used to prepare this checkout, the check was blocked by the host's GPU/NVML
policy and reported `torch.cuda.is_available() == false`. Re-run it on the
actual WSL2 + RTX 3090 host before starting the probe.

## Prepare the fixed NELL23K smoke input

The checked-in raw NELL23K source is under:

```text
grip-exp/data/raw_datasets/nell23k/
```

Prepare exactly the smoke sizes required by the experiment:

```bash
CONDA_ENV=guardenv \\
MAX_TRAIN_QUESTIONS=64 \\
MAX_VALIDATION_QUESTIONS=32 \\
MAX_TEST_QUESTIONS=64 \\
SEED=2026 \\
bash configs/prepare_nell23k.sh
```

The generated input is ignored by Git:

```text
grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json
```

## Model, adapters, and cache

The default runner expects:

- a local Qwen2.5-0.5B cache under `grip-exp/model_cache/` (or the path in
  `MODEL_CACHE_DIR`);
- the existing parent diagnostic run at `PARENT_RUN`, containing:
  `train_k1/adapters/nell23k/` and `train_k2/adapters/nell23k/`;
- CUDA-visible RTX 3090 memory.

The wrapper first checks the project cache and then automatically discovers a
standard Hugging Face cache under `$HOME/.cache/huggingface/hub/`; alternatively,
set `MODEL_PATH` to a complete local Transformers directory. If neither exists,
download the model from ModelScope using `guardenv`:

```bash
cd grip-exp
conda run --no-capture-output -n guardenv python scripts/download_model.py \
  --model_name qwen-0.5b --model_cache_dir model_cache
cd ..
```

The downloader verifies all safetensor shards and resumes interrupted downloads.
The expected project-cache directory is
`grip-exp/model_cache/Qwen--Qwen2.5-0.5B-Instruct/`.

The probe enforces Qwen2.5-0.5B, NELL23K, evaluation recurrence `K=1`, ten
candidates per question, and batched candidate scoring. It selects
`length_normalized` versus `raw_sum` on validation only and freezes the selected
normalization before writing score-decoder predictions for all splits.

## Run and resume

Use a fresh immutable `RUN_ID` for every new run:

```bash
cd 08_experiments/FactorGRIP/v0_1_candidate_energy_probe_2026-08-30
export CONDA_ENV=guardenv
export RUN_ID=wsl3090_nell23k_candidate_energy_20260830_01
bash configs/run_nell23k_candidate_probe_wsl.sh
```

The wrapper records outer monotonic GPU wall time, appends console output under
`logs/`, and writes the run under:

```text
results/runs/<RUN_ID>/
```

An existing directory is never overwritten. To resume an interrupted run with
the same immutable ID:

```bash
RESUME=1 \\
CONDA_ENV=guardenv \\
bash configs/run_nell23k_candidate_probe_wsl.sh
```

Resume is accepted only when the existing `config.json` contains the same
`run_id`; otherwise the runner stops. Use a new ID rather than changing the
experimental configuration mid-run.

## Required artifacts and analysis

A completed run contains:

```text
results/runs/<RUN_ID>/
├── config.json
├── predictions.jsonl
├── candidate_scores.jsonl
├── analysis/
│   ├── decoder_accuracy.csv
│   ├── paired_decoder_effects.csv
│   ├── calibration.json
│   └── summary.json
├── environment.txt
└── run.log
```

Then inspect the frozen validation/test analysis:

```bash
RUN_DIR="$(cat results/LAST_CANDIDATE_ENERGY_RUN.txt)"
CONDA_ENV=guardenv conda run --no-capture-output -n "$CONDA_ENV" \\
  python grip-exp/scripts/analyze_candidate_energy.py \\
  --input_file "$RUN_DIR/predictions.jsonl" \\
  --output_dir "$RUN_DIR/analysis"
```

The analysis reports decoder accuracy, correct-versus-none paired deltas,
score-versus-free deltas, exact McNemar tests, bootstrap 95% confidence
intervals, candidate-order stability, and the pre-specified Go/Stop gate. The
single-graph limitation keeps the shuffled-adapter criterion unavailable, so a
reported Go decision must remain false unless that control is later supplied in
an explicitly separate experiment.

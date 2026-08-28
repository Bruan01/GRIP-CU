# RecurrentGRIP Fixed-Depth v1

RecurrentGRIP v1 keeps the Original GRIP runner unchanged and adds a separate
closed-book CLEGR path. A graph-specific LoRA is restricted to one decoder
layer, and that same layer is executed repeatedly with shared parameters.
Generation always uses `use_cache=False`.

## 0. Platform boundary

- macOS is the development and static-validation host;
- Windows WSL2 with an RTX 3090 24GB is the ML test and experiment host;
- formal runs use the version-local `grip-exp/.venv` created inside WSL2;
- formal launchers pass `--require_cuda true` and `--evaluation_device cuda`.

See `../ENVIRONMENT_WSL3090.md` for setup, transfer, smoke-test, and provenance
instructions.

## 1. Build the exact-hop StationShortestCount split

If `processed_test.json` is already available:

```bash
RAW_INPUT=/path/to/processed_test.json \
OUTPUT_FILE=/path/to/recurrent_station_shortest.json \
  bash ../configs/prepare_clegr.sh
```

If the official CLEGR PyG archive must be converted again, first install the
optional `torch-geometric` dependency in the WSL environment, then run:

```bash
.venv/bin/python data/raw_datasets/clegr/process.py
```

The converter preserves `question_type`, `question_group`, and
`question_subgroup` when those fields exist in the official PyG objects. The
split builder recovers both station endpoints, recomputes undirected BFS
shortest distance, and accepts a sample only when the official count label is
`max(shortest_distance - 1, 0)`.

## 2. Verify the WSL2 RTX 3090 environment

From the version root:

```bash
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
```

The second command runs CUDA hardware checks and the complete unit-test suite.
`test_adapter_scope.py` is the key PEFT check: it verifies that
`layers_to_transform` still matches `layers.INDEX.block.*` after wrapping the
executor layer.

## 3. Run the single-graph smoke test

```bash
CLEGR_INPUT=/path/to/recurrent_station_shortest.json \
  bash configs/run_qwen05b_smoke_wsl.sh
```

The smoke test uses one graph and `K=1,2`. It checks training, adapter
save/load, explicit CUDA placement for the reloaded evaluation model,
cache-free generation, and JSONL output. Shuffled-adapter evaluation is skipped
because the control requires at least two graph adapters.

## 4. Run the two-hour pilot

```bash
CLEGR_INPUT=/path/to/recurrent_station_shortest.json \
  bash configs/run_qwen05b_pilot.sh
```

The pilot trains every graph adapter once and then evaluates:

- `correct`: the graph's own adapter;
- `shuffled`: the next graph's adapter in a deterministic cycle;
- `none`: all adapters disabled.

The formal launcher uses Qwen2.5-0.5B, BF16, CUDA evaluation, a 120-minute soft
limit, a 180-minute Python hard limit, and a 180-minute WSL `timeout`. Every
invocation receives a new `results/runs/<RUN_ID>_qwen05b_pilot/` directory, so
runs and failures are not overwritten. Original GRIP and GRIP+More-QA remain
separate baseline runners, so the new path does not alter their default
behavior.

## 5. Summarize results

```bash
# Defaults to results/LAST_PILOT_RUN.txt
bash ../configs/analyze_pilot.sh

# Or analyze an explicit immutable run
RUN_DIR=/path/to/results/runs/RUN_ID_qwen05b_pilot bash ../configs/analyze_pilot.sh
```

The summary reports accuracy by true hop, recurrence depth, and adapter
control, plus the Spearman correlation between true hop and the smallest
successful recurrence depth. Questions never solved at any tested `K` are
reported separately and excluded from the correlation.

# Remote WSL runbook

Run on the existing WSL2 + RTX 3090 environment after pulling the commit.
The server should not edit source files; all code changes are already in the
Git commit.

## 1. Pull and verify

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU
git fetch origin
git checkout wsl/nell23k-smoke-20260829
git pull --ff-only origin wsl/nell23k-smoke-20260829
git rev-parse HEAD
```

The reported commit must be the commit that added this experiment. Keep any
unrelated untracked files untouched.

## 2. Preflight

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU
source /home/kieran/miniconda3/etc/profile.d/conda.sh
conda activate guardenv
python --version
python - <<'PY'
import torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY

cd 13_base_method/grip-exp
python - <<'PY'
from constants import HF_DECODER_ONLY_LLMS, MODELSCOPE_DECODER_ONLY_LLMS
assert HF_DECODER_ONLY_LLMS['qwen-0.5b'] == 'Qwen/Qwen2.5-0.5B-Instruct'
assert MODELSCOPE_DECODER_ONLY_LLMS['qwen-0.5b'] == 'Qwen/Qwen2.5-0.5B-Instruct'
print('QWEN05B_MAPPING_READY')
PY
python -m py_compile constants.py models/ft_models/hf.py scripts/run_grip.py scripts/mp_wrapper.py
bash scripts/run_nell23k_qwen05b.sh --help
```

Confirm that the model path is complete. The known server snapshot layout is:

```text
/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<SNAPSHOT>
```

The wrapper discovers it automatically. If discovery chooses the wrong
cache, set `MODEL_PATH` explicitly.

## 3. Smoke run (optional)

A smoke run checks loading and command wiring only:

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU
export PYTHON_EXECUTABLE=/home/kieran/miniconda3/envs/guardenv/bin/python
NUM_TEST=16 \
RUN_ID=wsl3090_grip_qwen05b_smoke_20260906_01 \
MODEL_PATH=/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<SNAPSHOT> \
bash 08_experiments/GRIPBaseline/v0_1_qwen05b_nell23k_2026-09-06/configs/run_qwen05b_nell23k_wsl.sh
```

Do not use the smoke number as the formal baseline.

## 4. Formal full-test run

Use a new stable `RUN_ID`; omit `NUM_TEST`:

```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU
export PYTHON_EXECUTABLE=/home/kieran/miniconda3/envs/guardenv/bin/python
export MODEL_PATH=/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<SNAPSHOT>
export TASK_GENERATOR_MODEL_NAME=qwen-7b
export RUN_ID=wsl3090_grip_qwen05b_full_20260906_01
bash 08_experiments/GRIPBaseline/v0_1_qwen05b_nell23k_2026-09-06/configs/run_qwen05b_nell23k_wsl.sh
```

If interrupted, rerun with the same `RUN_ID` and:

```bash
bash 08_experiments/GRIPBaseline/v0_1_qwen05b_nell23k_2026-09-06/configs/run_qwen05b_nell23k_wsl.sh --resume-outputs
```

## 5. Required artifact collection

Return or archive at least:

```text
13_base_method/grip-exp/artifacts/logs/nell23k_qwen05b_*.log
13_base_method/grip-exp/artifacts/task_cache/<RUN_ID>/
13_base_method/grip-exp/outputs/grip_inf/nell23k/nell23k_grip_qwen05b_<RUN_ID>.json
```

Also record:

- `git rev-parse HEAD`;
- `sha256sum outputs/data/nell23k/processed_test.json`;
- test row count and prediction row count;
- model snapshot path and its `config.json`/weight file hashes;
- task generator model and backbone model separately;
- seed and all command-line arguments;
- final `em`, `f1`, and `hit` values;
- whether the run was full test or smoke;
- any OOM, NaN, restart, or partial-output event.

Do not open or use test answers for task generation, prompt construction, or
model selection. Do not change code on the server.

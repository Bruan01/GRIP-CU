# GRIP-Qwen2.5-0.5B NELL23K baseline

## Purpose

This version measures the original GRIP pipeline with a **Qwen2.5-0.5B-Instruct
backbone** on NELL23K. It is a size-matched baseline for the existing
Qwen2.5-0.5B mechanism experiments; it is not a replacement for the official
Qwen2.5-7B result.

The official task-generation setting is retained by default:

- backbone: `qwen-0.5b` / `Qwen/Qwen2.5-0.5B-Instruct`;
- task generator: `qwen-7b`;
- graph-free inference: `no_graph_context=True`;
- LoRA target modules: `down_proj up_proj gate_proj`;
- full `processed_test.json` unless `NUM_TEST` is explicitly set;
- metrics: `em f1 hit`.

The task generator and backbone are recorded separately in the runtime log.

## Implementation changes

- `13_base_method/grip-exp/constants.py` registers `qwen-0.5b` for Hugging
  Face and ModelScope aliases.
- `13_base_method/grip-exp/models/ft_models/hf.py` accepts an explicit local
  Transformers directory. This is required for the server's standard
  Hugging Face snapshot cache layout.
- `13_base_method/grip-exp/scripts/run_nell23k_qwen05b.sh` is the executable
  baseline entry. The `configs/` script here is only an experiment wrapper.

The original `run_nell23k_full.sh` remains the Qwen2.5-7B paper-aligned entry
and is not overwritten.

## Current status

```text
LOCAL_IMPLEMENTATION_READY_FOR_WSL_RUN
```

No accuracy is recorded here until the remote full-test run produces auditable
predictions, metrics, environment information, and the corresponding Git
commit.

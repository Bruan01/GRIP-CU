# GRIP paper protocol on one RTX 3090

The canonical settings are defined in `grip/paper_protocol.py`.  The paper
runner is the original per-graph `scripts/run_grip.py` path, not the
dataset-level task-file adapter.

## What is aligned

| Item | WN18RR | CLEGR-Reasoning |
|---|---:|---:|
| Context / reasoning QA / summarization | 8000 / 2000 / 6000 | 20 / 160 / 20 |
| LoRA rank / alpha | 24 / 48 | 16 / 32 |
| Stage 1 max epochs | 1 | 50 (minimum 5) |
| Stage 2 max epochs | 5 (minimum 1) | 50 (minimum 5) |
| Early-stop threshold | 0.15 | 0.40 |
| Main evaluation graph context | none | none |
| Main metric | Exact Match | Qwen2.5-32B LLM-as-Judge |

On a 24 GB RTX 3090, micro-batch size is set to 4 with gradient checkpointing;
WN18RR uses gradient accumulation 128 (the paper's exact `4 x 128` effective
batch).  CLEGR keeps the paper's `gather_batches=True`
semantics; its accumulation value is therefore controlled by the full task
dataset.  The paper used vLLM for task generation.  This WSL image has no
`nvcc`, so the shells disable only the optional FlashInfer sampler
(`VLLM_USE_FLASHINFER_SAMPLER=0`) and keep vLLM's prebuilt attention path.
The model, prompts, task counts, and training protocol are unchanged.  This
is a backend compatibility setting, not a scientific protocol change.

Run the WN18RR path with:

```bash
bash shells/qwen_grip_inf/wn18rr.sh
```

The repository's development split is 300/100/100 over a 500-graph source,
whereas the original GRIP preprocessing uses 20% train reference graphs and
the remaining 400 graphs for test.  The official shell now materializes and
uses `04_benchmark/data/clegr_reasoning/paper_split/` with a deterministic
seed-2026 manifest before requesting 400 test graphs.  The explicitly labelled
local split path remains available for a smoke/pipeline check:

```bash
bash shells/qwen_grip_inf/clegr_reasoning_3090_local_split.sh
```

Its output must remain a pilot/local-split result; it is not the paper-main
run.

## CLEGR judge

The paper uses Qwen2.5-32B because CLEGR answers can be semantically correct
while differing lexically from the reference string.  Exact Match only counts
normalized string identity, so it can undercount valid answers such as a
paraphrase, a reordered list, or a differently worded relation description.
The judge is instructed to return only `yes`/`no`, matching the original
metric implementation, and records raw judge outputs for audit.

The judge reads `13_base_method/grip-exp/.env` automatically.  The file is
ignored by Git and should remain local:

```bash
SILICONFLOW_API_KEY=your-key
```

Alternatively, an already-exported `SILICONFLOW_API_KEY` takes precedence.

Then judge the prediction JSONL:

```bash
/home/kieran/miniconda3/envs/guardenv/bin/python \
  scripts/judge_clegr_siliconflow.py \
  --input_file outputs/grip_inf/clegr_reasoning/clegr_reasoning_qwen_paper_3090.json \
  --output_file outputs/grip_inf/clegr_reasoning/clegr_reasoning_qwen_paper_3090_judge.jsonl \
  --model Qwen/Qwen2.5-32B-Instruct
```

The script caches by question/candidate/target/model/prompt hash and can be
rerun safely.  Its summary reports the strict paper score, cache hits, invalid
judge responses, and API call count without storing the credential.

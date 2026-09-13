#!/usr/bin/env bash
# Listed vs original GRIP on Qwen2.5-7B, paper LoRA/batch recipe.
#
# Same contrastive paradigm as run_listed_vs_b1.sh: shared Stage 1 graph
# storage, then B1 (generation only) vs listed InfoNCE.
# Training defaults to grip_nell23k_tasks.json (Qwen2.5-7B generated
# context + summarization + QA). Relation-like QA items sample 9 in-vocab
# negatives because those prompts have no official 10-way list. Val/test
# EM still uses the aligned NELL23K relation-prediction split.
#
# SCALE=smoke|pilot. One 24GB 3090; Stage 2 runs sequentially.
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-qwen-7b}"
export SCALE="${SCALE:-smoke}"
# One 24GB 3090: do not fork Stage 2 onto a second GPU.
export PARALLEL_S2="${PARALLEL_S2:-0}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_listed_vs_b1.sh"

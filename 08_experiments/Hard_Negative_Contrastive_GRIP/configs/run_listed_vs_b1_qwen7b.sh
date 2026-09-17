#!/usr/bin/env bash
# Listed vs original GRIP on Qwen2.5-7B, paper LoRA/batch recipe.
#
# Same contrastive paradigm as run_listed_vs_b1.sh: shared Stage 1 graph
# storage, then B1 (generation only) vs listed InfoNCE.
# Training defaults to grip_nell23k_tasks.json (Qwen2.5-7B generated
# context + summarization + QA). Listed InfoNCE negatives default to the
# official 198 train-graph relations (process.py rule). Set
# LISTED_NEGATIVE_SOURCE=qa_vocab to reproduce the 20260913 370-relation
# QA-gold pool. Val/test EM still uses the aligned 10-way split.
#
# SCALE=smoke|pilot. One 24GB 3090; Stage 2 runs sequentially.
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-qwen-7b}"
export SCALE="${SCALE:-smoke}"
# One 24GB 3090: do not fork Stage 2 onto a second GPU.
export PARALLEL_S2="${PARALLEL_S2:-0}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_listed_vs_b1.sh"

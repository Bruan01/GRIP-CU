#!/usr/bin/env bash
# Listed vs original GRIP on Qwen2.5-7B, paper LoRA/batch recipe.
#
# Same contrastive paradigm as run_listed_vs_b1.sh: shared Stage 1 graph
# storage, then B1 (generation only) vs listed 10-way InfoNCE.
# Data stays the aligned NELL23K relation-prediction split because listed
# contrast needs the prompt's 10-way distractors. The paper's 16k generated
# context/reason/summary tasks do not carry those lists.
#
# SCALE=smoke|pilot. One 24GB 3090; Stage 2 runs sequentially.
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-qwen-7b}"
export SCALE="${SCALE:-smoke}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_listed_vs_b1.sh"

# WN18RR GRIP / Structure-Counterfactual-9 Run Plan

## Scope

This run is the WN18RR paper-protocol alignment run on one RTX 3090. It keeps
the official GRIP task file and no-graph-context evaluation protocol, then
compares the frozen GRIP B1 baseline with the structure-counterfactual-9
candidate InfoNCE fork.

## Fixed protocol

- task file: the complete WN18RR GRIP artifact (8,000 context, 2,000
  reasoning, 6,000 summarization generation budgets; 4,096-token training)
- graph context: disabled for the paper-main route
- train graph filter: train split only
- relation vocabulary: the 11 WN18RR train relations, with the first raw
  underscore removed to match task labels
- listed negatives: K=9, attached only to the 2,709 relation QA rows
- LoRA: r=24, alpha=48, down/up/gate projections
- Stage 1: one epoch, effective batch 512
- Stage 2: up to five epochs, effective batch 512
- dtype: bfloat16, gradient checkpointing enabled where required by the 24 GB
  RTX 3090
- primary paper metric: exact match on the official processed test split

## Resumability boundaries

1. baseline Stage 1 checkpoint directory
2. baseline Stage 2 checkpoint directory and final S1/B1 adapters
3. resumable B1 full-vocabulary candidate scores
4. immutable structure-counterfactual bank
5. CPU wiring gate
6. resumable listed InfoNCE training checkpoints and final adapter

Every boundary must leave a log, metadata/configuration, and a status artifact.
An interrupted step is resumed only from its own checkpoint/artifact; later
steps are not started until their required artifact is complete.

## Execution order

`preflight -> baseline -> score -> freeze -> wire -> train -> evaluate/report`

The CLEGR task-generation process is out of scope for this WN run and is not
used as a dependency.

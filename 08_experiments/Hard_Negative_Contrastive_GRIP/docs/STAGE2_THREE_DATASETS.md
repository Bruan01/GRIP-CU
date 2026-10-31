# Stage-2 search: NELL23K, FB15K237-2 and CoDEx-M

2026-10-31 efficiency and storage update:

- Training scores are registered once, with teacher weight/config hashes,
  task/train/mapping hashes, relation order, scoring protocol and dtype. Only
  compatible complete QA blocks may be reused for a smaller eligibility set;
  test scores cannot enter the training-score cache. Fully cached work does
  not load a GPU model. Historical files without verified identities remain
  available but are not silently imported into a new experiment.
- Greedy predictions are cached by immutable adapter, local base/tokenizer,
  decoder/prompt implementation and verified source cohort. A validation
  screen shares predictions with its exact full validation source, with IDs
  remapped by question and target. Candidate-list changes, another split,
  changed weights or decoder invalidate reuse. This does not alter EM or
  replace free generation with candidate reranking.
- Checksums reuse inode/size/mtime/ctime validation, avoiding repeated reads
  of unchanged multi-GB files across workers. The cache lives on the Linux
  filesystem at `/home/kieran/grip-cu-runs/shared_cache`.
- Periodic recovery checkpoints are saved every16 steps with only one latest
  checkpoint retained. Every rung forces its own save, including step8 and
  nonmultiple final steps. Resume still restores optimizer/scheduler/RNG state.
- Between GPU subprocesses, protect the global top two validation snapshots
  and the final selected adapter. Each unfinished trial keeps its latest
  complete recovery checkpoint; an incomplete newer folder cannot displace it.
  A redundant packaged live adapter can be removed only after its evaluated
  rung has a complete optimizer, scheduler, RNG and weight checkpoint.
  Remove lower-ranked evaluated snapshot weights
  and terminal trials' redundant optimizer/live-adapter binaries. Keep metrics,
  predictions, trainer-state JSON, configs, logs and cleanup manifests.
  Immutable Stage1/B1 and historical paper weights are outside this cleanup.
- `search/storage_status.json` reports retained binary bytes and free space;
  `storage_cleanup.jsonl` records actual removals. Before a new training
  subprocess, require30GiB free on its actual output filesystem. If necessary
  space cannot be reclaimed safely, report the failure rather than deleting
  active/selected assets or risking a full disk.

The running NELL scoring process was preserved during this rollout. Subsequent
worker/runner processes load these changes without restarting that GPU job.

2026-10-31 21:31 recovery: NELL finished2974/2974 GPU-scored QA successfully,
but its outer Bash process failed at EOF after the on-disk runner was edited
while the child was running. The completed588852 score rows were verified and
reused for CPU-only bank construction/wiring; GPU scoring was not repeated.
Future runner launches use a content-addressed, read-only Bash snapshot, with
the original HNG root supplied explicitly, so later source edits cannot change
an already running shell's unread tail.

For controller-only reloads during an existing baseline trainer, `--wait-stage`
accepts an inspected process record containing PID, start ticks, command hash,
dataset, stage and log. It observes that exact Stage2 trainer before acquiring
its inherited controller flock. It neither interrupts the trainer nor treats
a reused PID as the original job; final baseline assets are validated normally
before search resumes. The CoDEx trainer was preserved during the NELL repair.

Current user contract (2026-10-30): all three datasets must show measured
free-generation EM improvement, one training seed is sufficient, and
hyperparameter search is restricted to Stage 2. This phase uses seed 2026.
The campaign records success only when the selected full-test EM exceeds
both its matched B1 and the Qwen-7B paper reference (NELL 87.74%, FB 88.75%,
CoDEx-M 94.42%, GRIP arXiv v2). A search budget ending without those results
is explicitly recorded as unsuccessful, not as a completed research goal.

Launch or inspect:

```bash
bash 08_experiments/Hard_Negative_Contrastive_GRIP/configs/run_stage2_three_datasets.sh
tmux attach -t grip_stage2_three_datasets
```

Live status and contracts are under
`results/hpo/stage2_three_dataset_20261030/STATUS.md` and `status.json`.
The launcher uses guardenv, explicit `PYTHONPATH`, an independent tmux
session and an exclusive controller lock. The previous FB prerequisite
process is adopted without interruption; its one-time Stage-1 restoration
was already in progress before this Stage-2 campaign. The old controller
is retired, and its launcher forwards to the new one.

All new baseline-resume commands set `--stage1_epochs 0`. A missing Stage-1
adapter causes an explicit asset error; it never triggers a new Stage-1
training or search. NELL reuses its complete cached Stage-1 and paired
generation baseline from `stage2_memory_retention/nell_r1`. CoDEx reuses its
complete Stage-1 and resumes Stage 2 from its existing optimizer checkpoint.
FB reuses the fixed adapter produced by the already-running prerequisite
job. Stage-1 hashes are checked before/after Stage-2 jobs and search turns.

Each dataset has 12 declared configurations. The queue alternates datasets,
adding two configurations per turn and increasing the optimizer-step cap
from 32 to 64, 128 and the fixed full-budget horizon. Partially evaluated
trials remain RUNNING in Optuna and resume the same optimizer checkpoints.
Pruning starts at step 32. Search changes candidate loss weight,
contrastive temperature, Stage-2 learning rate and weight decay. Candidate
count, Stage-1 weights, LoRA shape, task files, decoder and seed stay fixed.
The validation-selected checkpoint's training horizon is also a Stage-2
choice. It is not forced through extra training before test evaluation.

Search uses one fixed 1,000-question validation sample per dataset. After
the declared search, two checkpoints advance to full validation. B1 and one
selected SC-CL checkpoint are tested only after freezing the decision;
test scores are not inputs to the search or pruning. NELL had prior test
inspection in earlier campaigns, and its task-generation metadata also
records validation few-shot use; these facts are disclosed in the contract.
All results remain single-seed observations under the local protocols.

Generation QA text, graph context and full test files are unchanged by this
campaign. FB retains the previously documented repaired task copy. An
explicit, task/train-hash-bound eligibility manifest limits candidate loss
to QA whose entity pair and gold relation are verified against train.
Other QA retain their original generation loss and have no candidate loss.
Current eligible counts are NELL 2,974, CoDEx 2,715 and FB 3,109. No test or
validation labels are used in this eligibility decision. Train scoring,
bank freezing, wiring and training all use the same manifest.

Large search checkpoints and frozen candidate adapters are stored on the
Linux filesystem at `/home/kieran/grip-cu-runs/stage2_three_dataset_20261030/`
through experiment-directory symlinks. Predictions, configs, trial tables,
weight hashes and logs remain inspectable. Dataset failures are recorded
explicitly and do not silently prevent ready datasets from running.

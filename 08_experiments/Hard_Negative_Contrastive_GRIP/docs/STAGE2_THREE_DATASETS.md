# Stage-2 search: NELL23K, FB15K237-2 and CoDEx-M

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

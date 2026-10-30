# NELL23K / FB15K237-2 background campaign

Launch from the repository root:

```bash
bash 08_experiments/Hard_Negative_Contrastive_GRIP/configs/run_nell_fb_hpo_guard.sh
```

The launcher creates the independent tmux session `grip_nell_fb_hpo_guard`.
It uses `/home/kieran/miniconda3/envs/guardenv/bin/python` and explicitly supplies
the baseline repository and method directories in `PYTHONPATH`. Chat/terminal
disconnection does not stop this session. WSL shutdown or a host reboot does.

Live artifacts are under `results/hpo/nell_fb_guard_20261029/`:

- `STATUS.md`, `status.json`: progress, GPU, processes, latest log and measured results, updated every minute.
- `events.jsonl`, `supervisor.log`, `logs/`: commands, stage transitions and failures.
- `environment.json`, `plan.json`: actual runtime versions and bounded execution plan.
- `fb15k237_2/input_audit.json`, `task_repairs.json`: FB training-input checks and exact repair records.
- `fb15k237_2/search/selection.json`: validation-only checkpoint decision before test.
- `fb15k237_2/search/final_result.json` or `no_go.json`: measured selected-model result or failed promotion.

The controller first observes the existing `nell_step12_extension` tmux chain.
It waits for other GPU processes to exit before starting FB. It never kills an
unrelated GPU job. A file lock prevents duplicate controllers and remains held
by its direct child if the controller exits unexpectedly.

FB prerequisites are rebuilt from the current full task file. Stage 1 and B1
are distinct adapters (rank 24, alpha 48; one context epoch, at most five QA
epochs; effective batch 512). B1 is used for train-only candidate scores and the
validation reference. Every SC-CL trial starts from the same Stage-1 adapter.

The input preflight found that FB needs an ID-to-description mapping and that
464 of its 3,109 relation QA examples cannot be verified as matching a training
entity pair using the exact parser. The campaign retains the original task and
derives a repaired copy. For each unresolved QA, it generates a new canonical
relation question from a real train edge with the same gold relation. Selection
is deterministic: SHA256(seed, QA ID, gold) indexes the sorted train-pair pool.
This regenerates an example; it does not claim to recover the old question's
intended entities. Other QA and all context samples remain identical. B1 and
SC-CL both consume the derived copy. All 3,109 relation QA must then resolve to
a train pair with the gold relation. Validation/test data are not used in this
repair. This data change must be disclosed with results.

FB search is bounded to six Optuna configurations and optimizer-step rungs
8/16/32, with pruning starting at step 16. The exact default SC-CL setting and
the previous NELL proposal are protected initial configurations. Search uses a
fixed validation sample of 1,000 questions. Two checkpoints advance to full
validation against B1. The best full-validation checkpoint is frozen before
testing and is not trained further. If it fails to improve on B1 validation EM,
the SC-CL test is skipped. Only one selected SC-CL checkpoint and matched B1 are
tested. Weight/data hashes guard evaluation caches; trial state and checkpoints
allow interrupted searches to resume.

Paper references are Qwen-7B free-generation EM 87.74% (NELL) and 88.75% (FB),
from [GRIP arXiv v2, Tables 2–3](https://arxiv.org/html/2511.07457v2).
Local NELL has 4,943 questions; FB uses the existing fixed 10,000-question test.
Any comparison must state these local protocols, the FB task repair and the
single seed. Prior NELL test results have already been inspected, so further
NELL follow-ups are adaptive exploration rather than an untouched final holdout.
No score improvement is guaranteed by the search.

On a stage error the controller records the failing command and stops. Inspect
the stage log, repair the cause, then rerun the launcher to resume completed
stages and resumable Trainer/Optuna state. It does not repeatedly relaunch a
failing training command or delete checkpoints automatically.

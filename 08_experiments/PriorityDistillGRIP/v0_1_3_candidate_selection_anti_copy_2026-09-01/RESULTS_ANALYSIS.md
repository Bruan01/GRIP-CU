# v0.1.3 Results Analysis

This file will be generated after the WSL GPU runs. The primary result is the validation-selected graph-free test accuracy for each protocol. Do not select checkpoints using test labels.

Interpretation order:

1. Compare `candidate_selection_two_stage` against `oracle_two_stage` to isolate the cost/benefit of requiring candidate matching.
2. Compare `candidate_selection_anti_copy` against `candidate_selection_two_stage` to detect terminal-copy dependence.
3. Compare `candidate_selection_anti_copy_replay` against `candidate_selection_anti_copy` to measure Stage-2 forgetting.
4. Compare all protocols against `direct_answer_only`.

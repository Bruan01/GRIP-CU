# Patch Manifest — PriorityDistill-GRIP v0.1.6

This version is an independent experiment snapshot and does not modify the original GRIP implementation.

Added or changed:

- graph-free joint trace + answer supervision;
- terminal-masked trace target with a separate final-answer target;
- checkpoint audit for train, masked gold trace and visible-terminal oracle conditions;
- teacher-forced candidate answer ranking diagnostic;
- corrected checkpoint-curve evaluation with `max_new_tokens=80`;
- tqdm progress bars for training and evaluation;
- static tests and setup validation for the v0.1.6 protocol.

Do not commit local model caches or checkpoint weights (`adapter_model.pt`).

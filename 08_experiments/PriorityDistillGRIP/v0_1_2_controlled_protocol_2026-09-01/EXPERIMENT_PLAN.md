# Experiment Plan — v0.1.2 Controlled Protocol

## Question

Does the gold-path first stage improve graph-free answering beyond a direct answer-only LoRA baseline when the total training budget and evaluation protocol are controlled?

## Required runs

1. `direct_answer_only`
2. `oracle_two_stage`

Run the same seed first. If the difference is promising, repeat both protocols with pre-registered seeds 43 and 44.

## Pass criteria

Do not implement a learned prioritizer until oracle two-stage beats direct answer-only on validation-selected held-out test accuracy across multiple seeds, and the gain survives anti-copy diagnostics.

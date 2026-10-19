# Literature review for ErrorMemory-9 + InfoNCE

Updated: 2026-10-19

## Scope

This note supports one narrow extension: add a candidate-level InfoNCE term to
the original GRIP relation-learning stage and construct its nine negatives
from frozen train-only GRIP errors plus candidate score violations. It does
not study a new executor, recursion, or a general KG negative-sampling theory.

## Baseline mechanism

GRIP (Feng et al., arXiv:2511.07457 / KDD 2026) stores graph information in a
graph-specific LoRA adapter and trains graph-context/QA generation. Its
relation task is expressed as a candidate-list QA prompt. The project keeps
that generation objective and adapter path unchanged; only the candidate
relation learning stage receives an auxiliary InfoNCE loss.

## Method precedents

- SimKGC (ACL 2022) establishes InfoNCE, in-batch negatives and memory-based
  negatives for language-model KGC. It motivates the candidate decision-set
  objective, not the ErrorMemory construction.
- SANS (EMNLP 2020) motivates selecting structurally meaningful KG negatives;
  here the conservative analogue is removing known train facts for the same
  entity pair.
- RotatE (ICLR 2019) and self-adversarial sampling motivate using current
  model scores to emphasize difficult candidates.
- Generative Hard Negative Mining (Findings ACL 2023) motivates treating
  model-generated mistakes as useful hard-negative evidence.
- ProGCL (ICML 2022) motivates combining hardness with false-negative risk;
  hence ErrorMemory-9 keeps a uniform floor and audits known positives.
- The negative-sampling survey by Madushanka and Ichise (arXiv:2402.19195)
  motivates explicit provenance and the open-world caveat.

## Narrow claim boundary

The literature already contains generic hard negatives, model-generated
negatives, adaptive hardness and contrastive KGC. The defensible local claim
is narrower: a frozen original-GRIP, gold-conditioned error memory plus score
violation can be tested as a train-only candidate sampler under a matched
Random-9 protocol. Any improvement must be demonstrated, not assumed.

## Primary sources

- GRIP: https://arxiv.org/abs/2511.07457
- SimKGC: https://aclanthology.org/2022.acl-long.295/
- SANS: https://aclanthology.org/2020.emnlp-main.492/
- RotatE: https://arxiv.org/abs/1902.10197
- Generative hard negatives: https://aclanthology.org/2023.findings-acl.362/
- ProGCL: https://proceedings.mlr.press/v162/xia22b.html
- KG negative-sampling review: https://arxiv.org/abs/2402.19195

## Experimental implication

The first comparison is original GRIP/B1, Random-9 + InfoNCE, and
ErrorMemory-9 + InfoNCE. Report free-generation EM alongside closed-set
relation metrics, wrong-in-list/OOV slices, false-negative audits, sampler
coverage, seed, commit, adapter provenance and the frozen bank hashes.

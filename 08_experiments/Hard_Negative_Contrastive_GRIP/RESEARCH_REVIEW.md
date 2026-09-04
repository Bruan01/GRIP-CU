# Literature Review

Updated: 2026-09-02

## Scope and search boundary

The search covered the GRIP paper and repository-facing description, knowledge graph completion (KGC) negative sampling, structure-aware negatives, hard-negative mining, and contrastive learning for language-model-based KGC. The goal was to identify direct overlap with the proposed combination, not to claim an exhaustive systematic review.

## Closest GRIP work

### GRIP: In-Parameter Graph Reasoning through Fine-Tuning Large Language Models

Feng et al., arXiv:2511.07457, 2025; KDD 2026 version listed by the authors.

GRIP injects a graph into lightweight LoRA parameters. Its training pipeline separates graph context memorization/summarization from context and reasoning QA. Inference uses the graph-specific adapter without passing the original graph context. The KGC setup casts relation prediction as a 10-way classification-style QA task and evaluates knowledge graphs including FB15K-237, WN18RR, CoDEx-M, and NELL23K.

The published objective is generation likelihood for context, summaries, and QA. The paper discusses task sampling and LoRA placement/rank, but the method description does not introduce an explicit candidate-level ranking loss over hard negatives or an adapter identity contrastive term.

Local code confirmation: the current GRIP/RecurrentGRIP path builds chat-formatted QA strings with one answer, then trains through the standard language-model loss in `grip/training/train.py`. The existing RecurrentGRIP evaluation already supports `correct`, `shuffled`, and `none` adapter controls, which makes adapter-level diagnostics feasible.

Sources:

- https://arxiv.org/abs/2511.07457
- https://arxiv.org/html/2511.07457v1

## Structure-aware KG negatives

### SANS: Structure Aware Negative Sampling in Knowledge Graphs

Ahrabian et al., EMNLP 2020, ACL Anthology ID `2020.emnlp-main.492`.

SANS restricts corrupted entities to a head or tail entity's k-hop neighborhood. The central insight is that neighborhood candidates are structurally and semantically more meaningful than uniform corruptions. It is inexpensive and does not require an auxiliary model or adversarial optimization.

Relation corruption is outside the core SANS idea. For this project, SANS motivates the same-side entity negative, while relation negatives and path-structure negatives extend the candidate taxonomy to the relation-prediction QA setting used by GRIP.

Source:

- https://aclanthology.org/2020.emnlp-main.492

### Negative sampling survey

Madushanka and Ichise, *Negative Sampling in Knowledge Graph Representation Learning: A Review*, arXiv:2402.19195.

The survey distinguishes random, probabilistic, external-model, auxiliary-data, dynamic, adversarial, self-adversarial, and mixing approaches. It emphasizes the open-world problem: an unobserved triple can be missing rather than false. This is a critical design constraint here. Candidate generation must exclude all known positives available to the split and report that this filter is conservative rather than a proof of falsity.

Source:

- https://arxiv.org/html/2402.19195v1

## Contrastive KGC and hard-negative mining

### SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models

Wang et al., ACL 2022.

SimKGC uses InfoNCE with in-batch negatives, a pre-batch memory of earlier negatives, and self-negatives. It shows that text-based KGC benefits from an efficient contrastive objective and that self-negatives provide a simple hard-negative mechanism.

The overlap is the use of a candidate-level contrastive objective with a language model. The difference is that SimKGC is a text-based entity representation/KGC model, whereas the proposed project preserves GRIP's graph-specific LoRA internalization and adds explicit graph-structure candidate families plus an adapter condition.

Source:

- https://aclanthology.org/2022.acl-long.295

### Improving Knowledge Graph Completion with Generative Hard Negative Mining

Qiao et al., Findings of ACL 2023.

This work generates entity negatives from the same decoding distribution as the anchor and uses a self-information-enhanced contrastive strategy to increase semantic closeness and diversity. It is the closest precedent for language-model-generated hard negatives in KGC.

The proposed project does not use a second sequence-to-sequence generator in the first version. It uses deterministic graph-derived candidate families, optionally ranked by the current GRIP model, to isolate whether structural hardness itself helps. A later extension can compare generated negatives under the same protocol.

Source:

- https://aclanthology.org/2023.findings-acl.362

### RotatE and self-adversarial negative sampling

Sun et al., ICLR 2019.

RotatE introduced self-adversarial negative sampling for KGE: the current model upweights high-scoring corrupted triples. This supports a dynamic selection stage but also highlights the need to monitor false negatives. The proposed implementation therefore separates candidate construction from optional score-based selection and records the candidate provenance.

Source:

- https://arxiv.org/abs/1902.10197

## Graph contrastive learning caution

### ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning

Xia et al., ICML 2022.

ProGCL argues that similarity-only hard-negative selection in graphs can select false negatives because message passing makes same-class samples similar. It combines similarity with an estimated probability that a negative is a true negative.

For this project, the direct implication is methodological: the hardest candidate by model score must not be treated as automatically correct. We will retain known-positive filtering, cap the score-based selection, log candidate families, and report false-negative audits where additional validation/test facts are available.

Source:

- https://arxiv.org/abs/2110.02027

## Related contrastive KG representation work

### KGE-CL: Contrastive Learning of Tensor Decomposition Based Knowledge Graph Embeddings

Xu et al., arXiv:2112.04871.

KGE-CL adds contrastive terms to tensor-decomposition KGE to bring related entities and entity-relation couples closer. It operates on learned graph embeddings rather than a generative LLM with graph knowledge stored in LoRA.

Source:

- https://arxiv.org/abs/2112.04871

## Overlap assessment

I found no paper in the searched sources that describes the exact combination of:

1. GRIP-style graph knowledge internalized in a graph-specific LoRA adapter;
2. explicit generation QA retained as the primary objective;
3. three graph-derived hard-negative families for relation prediction: same-head wrong relation, same-relation wrong tail, and path-structure-similar wrong endpoint; and
4. a second contrastive signal that correct adapter scores exceed disabled or perturbed adapter scores.

This is evidence of a plausible research gap, not proof of no prior work. The closest components are independently established by GRIP, SANS, SimKGC, generative KGC hard-negative mining, RotatE, and ProGCL. The paper claim should therefore be framed as a structure-aware contrastive extension of GRIP, with an explicit ablation against each prior-inspired component.

## Open risks

- Knowledge graphs are incomplete. A candidate absent from the training graph may be a held-out positive.
- Relation names and entity names may tokenize differently, so sequence likelihood is not automatically comparable without length normalization.
- A ranking loss can improve candidate ordering while hurting exact generated answers; both must be reported.
- Adapter-vs-none contrast is a diagnostic or auxiliary objective only if both conditions are computed on the same prompt and the disabled/perturbed adapter is treated as a negative view, not as a factual answer.
- The existing NELL23K setup contains one graph. A shuffled-adapter control needs at least two trained graphs; adapter identity claims should be tested on multi-graph data such as CLEGR before being generalized.

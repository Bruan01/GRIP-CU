# First Experiment Plan

Updated: 2026-09-02

## 1. Falsifiable question

Two questions, answered in order:

- **Method**: does adding structure-aware candidate discrimination to GRIP improve relation prediction without reducing generation quality?
- **Mechanism**: does the graph-specific LoRA adapter, rather than the base language model prior, carry the graph knowledge that drives the prediction?

The method question (H1) is the necessary condition; the mechanism question (H4) is the paper's differentiated claim.

## 2. Hypotheses

- H1: `GRIP + contrastive` improves candidate MRR and Hits@1 over Original GRIP at matched generation training budget.
- H2: tail-range-overlapping and path-local relation negatives are harder than uniform relation negatives, measured by baseline score and candidate rank.
- H3: the combined three-family objective improves exact match/accuracy more consistently than any single family.
- H4: an adapter auxiliary term increases the score margin between the correct graph adapter and disabled/shuffled adapters on the same question; equivalently, the graph-specific adapter — not the base LM prior — carries the graph knowledge.
- H5: the improvement is not explained by additional QA examples or extra optimization steps.

H4 is the differentiated, mechanism-level claim. H1 must hold before H4 is interpretable; a failure of either is a valid, reportable negative result and blocks expansion to larger sweeps.

## 3. Task formulation

For a graph-specific query `q` with correct answer `a+`, construct candidate answers `A(q) = {a+, a-1, ..., a-k}`. The model continues to generate the answer using the original GRIP objective. In addition, score the normalized answer continuation for each candidate and optimize:

`L_total = L_generation + lambda_candidate * L_candidate + lambda_adapter * L_adapter`

where:

- `L_generation` is the existing causal language-model loss;
- `L_candidate` is InfoNCE or margin ranking over the correct continuation and filtered candidate continuations;
- `L_adapter` contrasts the correct graph adapter against disabled or shuffled adapter views on the same prompt.

The first implementation uses a candidate-level API rather than changing the base model. This isolates candidate construction and loss correctness before adding a custom Trainer/model forward pass.

## 4. Negative families

NELL23K is relation prediction: the answer is a relation `r+` for a queried entity pair `(h, t)`. Every negative is therefore a **wrong relation** `r-` whose corruption `(h, r-, t)` is absent from all known train/validation/test triples.

### N0: Random relation

Uniformly sample `r- != r+` with a fixed seed, excluding `(h, r-, t) in known`. Low-structure control.

### N1: Uniform relation

Deterministically take the first valid relations in sorted order. This is the relation-level control that separates "the contrastive signal itself" from "which negatives are used".

### N2: Tail-range relation

A relation `r-` whose observed tail set overlaps `r+`'s tail set. Such relations point to the same kind of entity as the positive and are therefore type-confusable.

### N3: Path-local relation

A relation observed on an edge incident to an entity within `k` hops of `h` in the undirected training graph. Candidates are ordered by graph distance then stable relation ID; `structural_distance` records the closest such edge. The default is `k=2`.

N2 and N3 are **legacy structure families**. On quick01 they almost never equal the model's actual wrong answer (in-list 6–7%, true OOV 0%). Keep them only as controls.

### N4: Listed relation (primary decision-set family)

The 9 non-gold options in the question's 10-way list. This is the closed set the model is asked to choose from. Val/test lists must come from the official GRIP processor (`process.py`, `numpy.random` after `seed=2026`), not RecurrentGRIP's per-subset RNG. Train questions have no paper list and keep the RecurrentGRIP 10-way.

Using the list the model actually saw covers **174/174** in-list generation errors. Official lists cover only **9/174** of the already-run quick01 dump, because that dump used RecurrentGRIP lists.

### N5: Surface relation

In-vocabulary near-form relations (long shared prefix / high string ratio). Targets the 70/334 errors that emit a real NELL relation outside the 10-way. Top-4 surface hits **17/70**.

### N6: Hallucinated relation

Out-of-vocabulary prefix-preserving strings. This matches the *class* of the 90/334 true-OOV errors, but an a-priori splice generator retrieves **0/90** exact strings. Do not treat N6 as a retrieval method for observed hallucinations. True OOV is a decoding problem (constrain generation to the 10-way, or mine the model's own rollouts), not a graph-structure negative family.

All in-vocabulary families share the same open-world caveat: `(h, r-, t)` absent from the graph is a conservative negative, not a proof of falsity. The known-fact filter and false-negative audit are mandatory.

## 5. Candidate scoring

For each candidate answer, compute the sum of token log-probabilities over the answer continuation, divided by the number of answer tokens. The question prefix must be identical across candidates. Do not compare raw sequence sums without length normalization.

For each query, record:

- positive and negative normalized log-likelihoods;
- candidate rank, MRR, Hits@1, and Hits@K;
- candidate family and graph distance;
- whether the candidate appears in train, validation, or test before filtering;
- whether the candidate was selected by the baseline model score;
- generation response and exact-match status.

## 6. Controls and ablations

Run in this order on NELL23K smoke data:

| ID | Variant | Purpose |
|---|---|---|
| B0 | Base LM / no adapter | Leakage and prior knowledge control |
| B1 | Original GRIP | Primary baseline |
| B2 | GRIP + random relation negatives | Contrastive-loss control |
| B3 | GRIP + N1 | Uniform-relation family |
| B4 | GRIP + N2 | Tail-range relation family |
| B5 | GRIP + N3 | Path-local relation family |
| B6 | GRIP + N1+N2+N3 | Full structure-aware objective |
| B7 | B6 without candidate loss | Extra data/compute control |
| B8 | B6 without adapter loss | Adapter-term ablation |
| B9 | B6 with adapter loss only | Adapter-term isolation |
| B10 | B6 with shuffled/none adapter control | Adapter identity diagnostic |

Keep model, tokenizer, graph context, question IDs, seed, QA count, LoRA rank, target modules, epochs, and evaluation prompts fixed. When the candidate loss adds forward passes, report wall time and peak memory and include a compute-matched control.

## 7. Data protocol

### NELL23K first

Use the existing RecurrentGRIP preparation protocol:

- train triples define the graph and training QA;
- validation and test triples remain evaluation questions;
- all train/validation/test triples are protected during candidate construction;
- 10-way candidates are retained for the existing relation-prediction format;
- val/test 10-way lists are rewritten to the official GRIP protocol (`scripts/align_official_nell23k_lists.py`); train lists stay RecurrentGRIP;
- one graph means shuffled-adapter evaluation is unavailable; use `none` and defer shuffled adapter claims.

The first smoke configuration should use 64/32/64 train/validation/test QA, one seed, Qwen2.5-0.5B, and a hard timeout of 45 minutes. Expand to 512/128/512 only after the candidate audit and baseline pass.

### Multi-graph follow-up

Use CLEGR or another multi-graph split to test shuffled adapters. For a controlled structure claim, use a task family with explicit shortest-path labels and disjoint train/test hop sets. Do not infer causal hop reasoning from NELL23K's diagnostic distances.

## 8. Success gates

Advance from smoke to pilot only if:

1. at least 95% of generated candidates are unique and pass the known-fact filter;
2. the three candidate families have non-zero counts on the selected subset;
3. candidate likelihoods and ranks are reproducible for the same seed;
4. Original GRIP reproduces its expected baseline behavior;
5. B6 improves candidate Hits@1 or MRR without a more than 2-point absolute drop in EM/Accuracy;
6. the adapter margin in B6 exceeds B8 or the result is explicitly reported as a failed adapter hypothesis;
7. no NaN, CUDA error, silent CPU fallback, or unbounded memory growth occurs.

## 9. Analysis and paper claim boundary

This is an **analysis + extension** of GRIP, not a new paradigm. The novelty is the combination: structure-aware relation negatives plus an adapter-identity contrastive term applied to graph-in-LoRA GRIP. Neither component is new in isolation (see `RESEARCH_REVIEW.md`); the contribution is the combination and the mechanism evidence it enables.

Primary (method) claim: structure-aware contrastive supervision is a useful extension for parameterized graph reasoning in GRIP under a controlled relation-prediction protocol.

Differentiated (mechanism) claim: the graph-specific adapter, rather than the base LM prior, carries the graph knowledge that drives relation prediction. This is supported when the correct adapter's continuation score exceeds the disabled/shuffled adapter views on the same prompt.

Mechanism claim requires all of:

- per-family hardness evidence;
- ablation evidence;
- adapter control evidence (multi-graph shuffled control for a strong claim);
- matched compute and QA budget;
- false-negative audit.

Do not claim universal graph reasoning improvement from a single NELL23K graph or from candidate metrics alone. Treat the false-negative audit and the multi-graph adapter control as explicit contributions, not appendices.

## 10. Execution order

1. Run the standalone unit tests in `tests/`.
2. Run candidate audit on a small synthetic graph and the prepared NELL23K subset.
3. Add candidate scoring to a copy of the existing GRIP evaluation path.
4. Implement `L_candidate` in a custom Trainer/model wrapper while preserving `L_generation`.
5. Add adapter views only after B6 is stable.
6. Run B0-B6, then B7-B10.
7. Expand to multi-graph data and three seeds only after the smoke gates pass.

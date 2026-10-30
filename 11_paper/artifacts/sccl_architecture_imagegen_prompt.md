# SC-CL architecture figure prompt and QA

Generator: built-in `image_gen` (model selection unavailable). Final asset: `../main/figures/sccl_architecture_imagegen.png`.

## Final prompt

Create a publication-quality, landscape, flat vector-like architecture figure on white for Structure-Counterfactual Contrastive Learning (SC-CL). Use restrained navy, teal and amber, aligned boxes, readable sans-serif labels, thin arrows and generous whitespace. Show four numbered stages: **TRAIN-ONLY INPUTS** (Training graph + questions; Frozen B1 train scores), **CANDIDATE SIGNALS** (Known-pair filter → Candidate pool Pq; Structure profile from training graph; B1 score violation from frozen B1 scores), **BANK CONSTRUCTION** (three signals → Uniform floor + softmax → Frozen bank, K=9; audit tag “train-only · unique · manifest”), and **GRIP STAGE 2** (Candidate InfoNCE plus Generation loss, labeled “original GRIP objective”, join into Trained model). Separate validation/test beneath a dashed “No validation/test feedback into bank” barrier; the trained model evaluates Free generation EM and Candidate ranking MRR / Hits. Do not draw feedback from evaluation into bank, entity corruption, verified false facts, results, gradients, shadows, or decorative art.

## Review of final image

- The frozen B1 branch ends at score violation; it does not feed the generation loss.
- The structure profile comes from the training graph; the candidate pool passes known-pair filtering.
- The bank is visibly frozen before Stage 2 and has $K=9$.
- Generation and candidate losses meet at the trained model; evaluation is separated below.
- English labels were visually checked at full resolution. No result values or unsupported truth claims appear.

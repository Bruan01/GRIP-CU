# Framework Figure Studio Pro v3.2.15c — SC-CL figure contract

Source: `figure_contract.md`, manuscript formulation/method. This is one source-grounded formal candidate, scoped to the user's request for one saved figure from this skill.

## Semantic graph

| Node | Visible treatment | Source |
|---|---|---|
| Training graph and questions | left source card | method: train-only relation vocabulary and profiles |
| Frozen B1 train scores | left source card | method: score violation on training questions |
| Known-pair filter | filter gate | formulation: $P_q$ excludes gold and train-known relations |
| Structure profile | compact signal chip | method: $s_{\mathrm{str}}$ |
| B1 score violation | compact signal chip | method: $V_{\mathrm{B1}}$ |
| Mixture sampler | central process box | method: uniform floor plus softmax |
| Frozen candidate bank | focal artifact, $K=9$ tag | method: freeze before Stage 2 |
| Candidate InfoNCE | right training branch | formulation: $\mathcal{L}_{\mathrm{cand}}$ |
| Original generation loss | right training branch | formulation: $\mathcal{L}_{\mathrm{gen}}$ |
| Trained GRIP model | right merge output | formulation: joint objective |
| Free generation / candidate ranking | evaluation strip | formulation: separate metrics |

Edges: graph → filter/profile; frozen B1 → violation; filtered pool/profile/violation → sampler; sampler → frozen bank; bank → Candidate InfoNCE; Candidate InfoNCE and original generation loss → trained model; trained model → the two evaluation readouts. The original generation loss is not fed by frozen B1 scores. Evaluation has no reverse edge to train-only modules or bank.

Visible text whitelist: `TRAIN ONLY`, `Training graph + questions`, `Frozen B1 scores`, `Known-pair filter`, `Structure profile`, `B1 score violation`, `Mixture sampler`, `Frozen bank`, `K=9`, `Candidate InfoNCE`, `Generation loss`, `Trained GRIP`, `EVALUATION ONLY`, `Free generation EM`, `Candidate ranking MRR / Hits`, `No feedback to bank`.

Visual render graph: three horizontal zones — sources/signals, hero bank, Stage 2 — plus a slim evaluation strip. $P_q$, $s_{\mathrm{str}}$, $V_{\mathrm{B1}}$ are edge or port tags if shown, never peer modules. Use one arrow per block-level relation, merging signal arrows before the sampler. No duplicate parallel process lanes. Style: ACM/IEEE/AAAI line-art schematic, readable at double-column width.

Negative constraints: no test-data arrow into sampling, no entity corruption, no "verified negatives", no invented metric values, no neural-network decorations, no code or IDs visible, no unrelated icons.

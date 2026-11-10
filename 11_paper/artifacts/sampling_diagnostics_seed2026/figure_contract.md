# Literature-guided sampling diagnostics: fixed contract

Backend: established guardenv Python/Matplotlib. CPU only; no retraining,
inference, negative resampling, or new hyperparameter selection.

Question: how do the actual K=9, seed=2026 negative banks change relation-profile
similarity and frozen B1 difficulty within each eligible training question?
This is a sampler diagnostic, not a distribution of final model predictions.

Cohort: exactly the previous four-dataset distribution audit (2974 / 2715 /
3109 / 2709 questions). WN18RR retains its historical masks and 55 unverified
entity pairs. CoDEx-M has no observed Uniform/quota bank; its baseline is an
analytic expectation over its legal candidate pool. Other controls use actual
completed banks. Pending frequency-only training is outside this cohort.

Measures: reconstruct the five-feature log1p relation-profile similarity from
frozen profiles; use cached pairwise_confusion as B1 violation/difficulty.
Within-query percentiles are (average rank - 0.5)/pool size; larger means more
similar/harder. Ties receive midranks. Each query has equal mass; actual banks
give each of nine slots equal weight. Analytic Uniform gives each legal
candidate mass 1/pool size. It is not a globally flat relation distribution.

Fixed bins: 10 equal intervals on [0,1], both raw signals and percentiles.
Joint maps: log2(P_Full(cell)/P_Uniform_expected(cell)); impossible cells are
masked, observed zeros have -infinity lift and use the lower color extension.
Common color range [-2,2]; no pseudocounts. Within-query rank ties imply the
Uniform histogram need not be flat, especially with only 9–10 WN candidates.

Figures: (1) hardness/structure percentile histograms in a four-dataset grid;
(2) joint selection-lift grid; (3) raw-edge vs negative relation probability
scatter with descriptive equality line; (4) completed FB single-signal control
mean-percentile fingerprints; (5–8) per-dataset raw signal and paired
question-level diagnostics. Marginal scatter imports already audited CSV data.
No mixture model or true/false negative labels: those labels are unavailable.
No power-law fit, temperature sweep, or embedding projection is invented.

Interpretation: describe measured selection shifts and heterogeneity. Joint
enrichment does not demonstrate causal interaction or downstream gains. Raw
edge matching is not the objective and is not required to be successful.

Deliverables: inspectable CSV + validation/provenance JSON, English vector
PDF/SVG, PNG previews, combined PDF, Chinese interpretation, LaTeX insertion
snippet. Render, inspect, revise; verify mass, feature reconstruction, exact
cohort/bank legality, source hashes, and export integrity before acceptance.

"""Candidate-Energy Probe for FactorGRIP v0.1.

Inference-only probe that reuses the RecurrentGRIP v1.1.1 trained NELL23K LoRA
adapters and compares three decoder types:

* ``free``        — unconstrained greedy generation (replicates v1.1.1 eval);
* ``constrained`` — trie-constrained greedy generation forcing the output to be
                    exactly one of the candidate relations;
* ``score``       — batched, length-normalised sequence log-likelihood ranking
                    over the ten candidates.

The probe never re-trains and never constructs an input from the target
relation. See ``SOURCE_BASELINE.md`` / ``PATCH_MANIFEST.md`` for provenance.
"""

from .constrained import (  # noqa: F401
  CandidateTrieLogitsProcessor,
  build_candidate_trie,
)
from .probe import (  # noqa: F401
  ADAPTER_CONTROLS,
  DECODER_TYPES,
  CandidateProbePrediction,
  run_candidate_energy_probe,
)
from .scoring import (  # noqa: F401
  compute_candidate_logprobs,
  score_candidates_batched,
)

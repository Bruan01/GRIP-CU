"""Small, framework-independent building blocks for the GRIP experiment."""

from .candidates import (
    Candidate,
    build_adjacency,
    generate_hallucinated_negatives,
    generate_hard_negatives,
    generate_random_negatives,
    known_triples,
    parse_listed_relations,
)
from .losses import adapter_contrastive_loss, candidate_infonce_loss, margin_ranking_loss
from .metrics import hits_at_k, reciprocal_rank, summarize_candidate_scores
from .scoring import (
    encode_without_specials,
    normalized_continuation_log_likelihood,
    pack_decision_set_rows,
    score_candidate_rows,
    score_candidates,
    unwrap_for_scoring,
)

__all__ = [
    "Candidate",
    "build_adjacency",
    "generate_hallucinated_negatives",
    "generate_hard_negatives",
    "generate_random_negatives",
    "known_triples",
    "parse_listed_relations",
    "adapter_contrastive_loss",
    "candidate_infonce_loss",
    "margin_ranking_loss",
    "hits_at_k",
    "reciprocal_rank",
    "summarize_candidate_scores",
    "normalized_continuation_log_likelihood",
    "pack_decision_set_rows",
    "score_candidate_rows",
    "score_candidates",
    "unwrap_for_scoring",
]

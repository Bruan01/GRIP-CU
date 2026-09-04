"""Small, framework-independent building blocks for the GRIP experiment."""

from .candidates import (
    Candidate,
    build_adjacency,
    generate_hard_negatives,
    generate_random_negatives,
    known_triples,
)
from .losses import adapter_contrastive_loss, candidate_infonce_loss, margin_ranking_loss
from .metrics import hits_at_k, reciprocal_rank, summarize_candidate_scores
from .scoring import normalized_continuation_log_likelihood

__all__ = [
    "Candidate",
    "build_adjacency",
    "generate_hard_negatives",
    "generate_random_negatives",
    "known_triples",
    "adapter_contrastive_loss",
    "candidate_infonce_loss",
    "margin_ranking_loss",
    "hits_at_k",
    "reciprocal_rank",
    "summarize_candidate_scores",
    "normalized_continuation_log_likelihood",
]

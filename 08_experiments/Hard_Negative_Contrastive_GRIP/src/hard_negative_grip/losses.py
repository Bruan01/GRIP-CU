from __future__ import annotations

import torch


def _check_scores(positive: torch.Tensor, negatives: torch.Tensor) -> None:
    if positive.ndim != 1:
        raise ValueError("positive scores must have shape [batch]")
    if negatives.ndim != 2 or negatives.shape[0] != positive.shape[0]:
        raise ValueError("negative scores must have shape [batch, num_negatives]")
    if negatives.shape[1] < 1:
        raise ValueError("at least one negative score is required")


def candidate_infonce_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """InfoNCE over a positive answer and its candidate negative answers."""
    _check_scores(positive_scores, negative_scores)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1) / temperature
    labels = torch.zeros(positive_scores.shape[0], dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def margin_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Average hinge loss requiring each positive to beat every negative."""
    _check_scores(positive_scores, negative_scores)
    if margin < 0:
        raise ValueError("margin must be non-negative")
    return torch.relu(margin - positive_scores.unsqueeze(1) + negative_scores).mean()


def adapter_contrastive_loss(
    correct_adapter_scores: torch.Tensor,
    negative_adapter_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Contrast correct graph adapter scores against none/shuffled views."""
    return candidate_infonce_loss(
        correct_adapter_scores,
        negative_adapter_scores,
        temperature=temperature,
    )

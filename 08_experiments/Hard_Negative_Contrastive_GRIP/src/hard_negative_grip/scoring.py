from __future__ import annotations

import torch


def normalized_continuation_log_likelihood(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score answer tokens under teacher forcing, normalized by answer length.

    ``logits[:, i]`` predicts ``input_ids[:, i + 1]``. Each row may have a
    different prompt length, but all rows must be padded to the same sequence
    length. ``prefix_lengths`` is the number of prefix (question) tokens in
    each row; ``sequence_lengths`` is the true token count of each row before
    padding. When ``sequence_lengths`` is ``None`` every row is assumed to be
    unpadded (its true length equals the padded length).

    Padding tokens after the true end of a row are excluded via
    ``sequence_lengths`` so they never contribute to the score.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if input_ids.ndim != 2 or input_ids.shape[:2] != logits.shape[:2]:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if prefix_lengths.ndim != 1 or prefix_lengths.shape[0] != input_ids.shape[0]:
        raise ValueError("prefix_lengths must have shape [batch]")

    batch, seq_len = input_ids.shape
    if sequence_lengths is None:
        sequence_lengths = torch.full(
            (batch,), seq_len, dtype=torch.long, device=input_ids.device
        )
    if sequence_lengths.ndim != 1 or sequence_lengths.shape[0] != batch:
        raise ValueError("sequence_lengths must have shape [batch]")
    if torch.any(sequence_lengths > seq_len):
        raise ValueError("sequence_lengths cannot exceed the padded sequence length")
    if torch.any(prefix_lengths < 1) or torch.any(prefix_lengths >= sequence_lengths):
        raise ValueError("each prefix must leave at least one answer token")

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_positions = torch.arange(seq_len - 1, device=input_ids.device).unsqueeze(0)
    answer_positions = (
        (token_positions >= (prefix_lengths - 1).unsqueeze(1))
        & (token_positions < (sequence_lengths - 1).unsqueeze(1))
    )
    token_scores = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_scores = token_scores.masked_fill(~answer_positions, 0.0)
    answer_lengths = answer_positions.sum(dim=1).to(token_scores.dtype)
    return token_scores.sum(dim=1) / answer_lengths

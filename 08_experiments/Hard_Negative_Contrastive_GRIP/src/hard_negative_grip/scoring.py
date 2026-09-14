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


def pack_token_rows(
    rows: list[list[int]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad token rows into a dense batch on ``device``."""
    if not rows:
        raise ValueError("rows must be non-empty")
    seq_lens = torch.tensor([len(row) for row in rows], dtype=torch.long, device=device)
    max_len = int(seq_lens.max().item())
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        end = len(row)
        input_ids[index, :end] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[index, :end] = 1
    return input_ids, attention_mask, seq_lens


def continuation_mean_log_likelihood(
    answer_logits: torch.Tensor,
    answer_token_ids: torch.Tensor,
    answer_lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean log-probability of already-sliced answer tokens."""
    if answer_logits.ndim != 3:
        raise ValueError("answer_logits must have shape [batch, answer, vocabulary]")
    if answer_token_ids.shape != answer_logits.shape[:2]:
        raise ValueError("answer_token_ids must match answer_logits in batch and length")
    if answer_lengths.ndim != 1 or answer_lengths.shape[0] != answer_logits.shape[0]:
        raise ValueError("answer_lengths must have shape [batch]")
    if torch.any(answer_lengths < 1) or torch.any(answer_lengths > answer_logits.shape[1]):
        raise ValueError("each answer must have at least one token and fit the padded length")

    log_probs = torch.log_softmax(answer_logits, dim=-1)
    token_scores = log_probs.gather(-1, answer_token_ids.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(answer_logits.shape[1], device=answer_logits.device).unsqueeze(0)
    mask = positions < answer_lengths.unsqueeze(1)
    token_scores = token_scores.masked_fill(~mask, 0.0)
    return token_scores.sum(dim=1) / answer_lengths.to(token_scores.dtype)


def slice_continuation_states(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep hidden states that predict continuation tokens, dropping the prefix."""
    answer_lengths = sequence_lengths - prefix_lengths
    max_answer = int(answer_lengths.max().item())
    batch, _, hidden_size = hidden.shape
    answer_hidden = hidden.new_zeros(batch, max_answer, hidden_size)
    answer_ids = input_ids.new_zeros(batch, max_answer)
    for index in range(batch):
        start = int(prefix_lengths[index].item()) - 1
        length = int(answer_lengths[index].item())
        answer_hidden[index, :length] = hidden[index, start : start + length]
        answer_ids[index, :length] = input_ids[index, start + 1 : start + 1 + length]
    return answer_hidden, answer_ids, answer_lengths

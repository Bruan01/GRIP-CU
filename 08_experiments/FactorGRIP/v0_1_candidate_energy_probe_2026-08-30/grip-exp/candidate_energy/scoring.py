"""Batched, length-normalised candidate sequence log-likelihood scoring.

Given a prompt (token ids produced by :class:`GRIPEvalDataset`) and a list of
candidate relation strings, this module scores each candidate by the
length-normalised log-likelihood of its formatted answer
``<answer>{candidate}</answer>`` continuation.

The prompt is identical for all ten candidates, so any boundary tokenisation
effect at the prompt/answer junction is a *constant* across candidates and does
not affect the argmax ranking — which is the only thing the probe uses the
score for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

_NEG_INF = -math.inf


@dataclass(frozen=True)
class CandidateScore:
    """Length-normalised log-likelihood for one candidate."""

    candidate: str
    sum_logprob: float
    num_tokens: int

    @property
    def norm_logprob(self) -> float:
        """Mean log-probability per answer token (higher is better)."""

        return self.sum_logprob / self.num_tokens if self.num_tokens else _NEG_INF


def _format_answer(tokenizer, candidate: str) -> list[int]:
    """Tokenise ``<answer>{candidate}</answer>`` without special tokens."""

    return tokenizer.encode(f"<answer>{candidate}</answer>", add_special_tokens=False)


def compute_candidate_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_spans: Sequence[tuple[int, int]],
) -> list[CandidateScore]:
    """Pure tensor function: gather per-candidate answer-token log-probs.

    Parameters
    ----------
    logits:
        ``[batch, seq_len, vocab]`` raw logits from a forward pass.
    input_ids:
        ``[batch, seq_len]`` token ids that produced *logits*.
    answer_spans:
        One ``(start, end)`` per batch row — the half-open range of answer
        tokens (prompt tokens are *before* ``start``).

    Returns one :class:`CandidateScore` per batch row.  Logits at position *j*
    predict the token at position *j + 1*; the answer token at absolute
    position *k* is therefore scored by ``log_softmax(logits[:, k-1, :])``.
    """

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch, seq, vocab] and input_ids must be [batch, seq]")
    if logits.shape[0] != len(answer_spans) or input_ids.shape[0] != len(answer_spans):
        raise ValueError("answer_spans length must match batch size")
    if logits.shape[1] != input_ids.shape[1]:
        raise ValueError("logits and input_ids sequence lengths must match")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    results: list[CandidateScore] = []
    for row, (start, end) in enumerate(answer_spans):
        if start < 1 or end > input_ids.shape[1] or end < start:
            raise ValueError(f"invalid answer span: {(start, end)}")
        num_tokens = end - start
        if num_tokens <= 0:
            results.append(
                CandidateScore(candidate="", sum_logprob=_NEG_INF, num_tokens=0)
            )
            continue
        answer_token_ids = input_ids[row, start:end]
        # positions [start-1 .. end-2] predict tokens [start .. end-1]
        predict_positions = torch.arange(start - 1, end - 1, device=logits.device)
        gathered = log_probs[row, predict_positions, :].gather(
            -1, answer_token_ids.unsqueeze(-1)
        )
        sum_lp = float(gathered.sum().item())
        results.append(
            CandidateScore(candidate="", sum_logprob=sum_lp, num_tokens=num_tokens)
        )
    return results


def score_candidates_batched(
    model,
    tokenizer,
    prompt_ids: list[int],
    candidates: list[str],
    device: torch.device,
    max_total_length: int = 1024,
) -> list[CandidateScore]:
    """Forward one batched pass scoring every candidate against the same prompt.

    Right-padding is used so that answer-token positions are simply
    ``[len(prompt), len(prompt) + len(answer))`` for every row — the causal
    mask plus ``attention_mask`` keeps pad tokens from influencing real
    positions.
    """

    if not candidates:
        return []

    sequences: list[list[int]] = []
    answer_spans: list[tuple[int, int]] = []
    prompt_len = len(prompt_ids)
    for candidate in candidates:
        answer_ids = _format_answer(tokenizer, candidate)
        full = list(prompt_ids) + answer_ids
        if len(full) > max_total_length:
            raise ValueError(
                f"prompt ({prompt_len}) + answer ({len(answer_ids)}) exceeds "
                f"max_total_length ({max_total_length})"
            )
        sequences.append(full)
        answer_spans.append((prompt_len, prompt_len + len(answer_ids)))

    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        model.train(was_training)

    scores = compute_candidate_logprobs(outputs.logits, input_ids, answer_spans)
    # Re-attach the original candidate strings (decode is bookkeeping only).
    return [
        CandidateScore(
            candidate=cand, sum_logprob=s.sum_logprob, num_tokens=s.num_tokens
        )
        for cand, s in zip(candidates, scores, strict=True)
    ]


def argmax_candidate(
    scores: list[CandidateScore], normalized: bool = True
) -> tuple[int, float]:
    """Return ``(index, score)`` of the best candidate.

    With *normalized* ``True`` (default) the length-normalised score is used;
    with ``False`` the raw sum is used (validation will pick the better).
    """

    if not scores:
        return -1, _NEG_INF
    key = (lambda s: s.norm_logprob) if normalized else (lambda s: s.sum_logprob)
    best_idx = max(range(len(scores)), key=lambda i: key(scores[i]))
    return best_idx, key(scores[best_idx])

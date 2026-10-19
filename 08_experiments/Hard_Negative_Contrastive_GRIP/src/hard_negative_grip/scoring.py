from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    """Right-pad token rows into a dense batch on ``device``.

    Padding is built on CPU, then moved once. Token ids and the mean-logprob
    definition are unchanged.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    lengths = [len(row) for row in rows]
    max_len = max(lengths)
    batch = len(rows)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for index, row in enumerate(rows):
        end = lengths[index]
        if end:
            input_ids[index, :end] = torch.as_tensor(row, dtype=torch.long)
            attention_mask[index, :end] = 1
    seq_lens = torch.tensor(lengths, dtype=torch.long)
    if device.type == "cpu":
        return input_ids, attention_mask, seq_lens
    return (
        input_ids.to(device, non_blocking=True),
        attention_mask.to(device, non_blocking=True),
        seq_lens.to(device, non_blocking=True),
    )


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


def unwrap_for_scoring(model, accelerator=None):
    """Avoid Accelerate wrapping, which materializes full fp32 logits."""
    if accelerator is not None:
        return accelerator.unwrap_model(model)
    return model


def unwrap_causal_lm(model, accelerator=None):
    """Return the CausalLM that owns ``model`` / ``lm_head``, if present."""
    inner = unwrap_for_scoring(model, accelerator)
    if hasattr(inner, "get_base_model"):
        inner = inner.get_base_model()
    if hasattr(inner, "model") and hasattr(inner, "lm_head"):
        return inner
    return None


def encode_without_specials(tokenizer, text: str) -> list[int]:
    """Tokenize a prefix or relation string without BOS/EOS/PAD."""
    encoded = tokenizer(text, add_special_tokens=False)
    ids = list(encoded["input_ids"])
    if ids:
        return ids
    unk = getattr(tokenizer, "unk_token_id", None)
    return [0 if unk is None else unk]


def encode_many_without_specials(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Batch-tokenize strings without specials; fall back to one-by-one.

    HuggingFace tokenizers accept a list and return the same ids as separate
    calls. Toy tokenizers in tests often accept only a string, so TypeError
    (or a non-batched return) falls back to ``encode_without_specials``.
    Padding is disabled so batched ids stay identical to per-string calls.
    """
    if not texts:
        return []
    try:
        encoded = tokenizer(list(texts), add_special_tokens=False, padding=False)
        ids_batch = encoded["input_ids"]
    except TypeError:
        return [encode_without_specials(tokenizer, text) for text in texts]
    if hasattr(ids_batch, "tolist"):
        ids_batch = ids_batch.tolist()
    if not isinstance(ids_batch, list) or len(ids_batch) != len(texts):
        return [encode_without_specials(tokenizer, text) for text in texts]
    if ids_batch and isinstance(ids_batch[0], int):
        return [encode_without_specials(tokenizer, text) for text in texts]
    unk = getattr(tokenizer, "unk_token_id", None)
    fallback = 0 if unk is None else unk
    rows: list[list[int]] = []
    for ids in ids_batch:
        ids = list(ids)
        rows.append(ids if ids else [fallback])
    return rows


def pack_decision_set_rows(
    tokenizer,
    prefix_text: str,
    relations: list[str],
) -> tuple[list[list[int]], list[int]]:
    """Tokenize one shared prefix plus each listed continuation."""
    if not prefix_text:
        raise ValueError("prefix_text must be non-empty")
    if not relations:
        raise ValueError("relations must be non-empty")
    prefix_ids = encode_without_specials(tokenizer, prefix_text)
    relation_ids = encode_many_without_specials(tokenizer, relations)
    rows = [prefix_ids + ids for ids in relation_ids]
    return rows, [len(prefix_ids)] * len(relations)


def score_packed_candidate_rows(
    scorer,
    tokenizer,
    rows: list[list[int]],
    prefix_lens: list[int],
    device: torch.device,
    accelerator=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed training scorer: mean log-probability of continuation tokens.

    For a row ``prefix + answer``, only the answer tokens participate. If the
    answer has ``L`` tokens, the score is

        (log p(a_1 | prefix) + ... + log p(a_L | prefix, a_<L)) / L

    Prompt / prefix tokens, right-padding, EOS, and ``</answer>`` are not part
    of the continuation, so they do not enter the mean. Temperature is not
    applied here; InfoNCE divides by temperature later.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    input_ids, attention_mask, seq_lens = pack_token_rows(
        rows, pad_id=int(pad_id), device=device
    )
    prefix_lengths = torch.tensor(prefix_lens, dtype=torch.long, device=device)
    answer_lens = seq_lens - prefix_lengths
    causal = unwrap_causal_lm(scorer, accelerator)
    if causal is not None:
        hidden = causal.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )[0]
        answer_hidden, answer_ids, sliced_lens = slice_continuation_states(
            hidden, input_ids, prefix_lengths, seq_lens
        )
        logits = causal.lm_head(answer_hidden)
        del hidden, answer_hidden
        scores = continuation_mean_log_likelihood(logits, answer_ids, sliced_lens)
        return scores, sliced_lens
    outputs = scorer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    scores = normalized_continuation_log_likelihood(
        outputs.logits,
        input_ids,
        prefix_lengths,
        seq_lens,
    )
    del outputs
    return scores, answer_lens


def score_candidate_rows(scorer, tokenizer, rows, prefix_lens, device, accelerator=None):
    """Score packed answer continuations with one forward pass.

    This is the training-path function used by listed InfoNCE. It returns only
    the mean log-probability tensor; token lengths are available from
    ``score_packed_candidate_rows`` / ``score_candidates``.
    """
    scores, _ = score_packed_candidate_rows(
        scorer,
        tokenizer,
        rows,
        prefix_lens,
        device,
        accelerator=accelerator,
    )
    return scores


def _normalize_score_inputs(
    questions: str | Sequence[str],
    candidate_answers: Sequence[str] | Sequence[Sequence[str]],
) -> tuple[list[str], list[list[str]]]:
    if isinstance(questions, str):
        question_list = [questions]
    else:
        question_list = [str(item) for item in questions]
    if not question_list:
        raise ValueError("questions must be non-empty")
    if not candidate_answers:
        raise ValueError("candidate_answers must be non-empty")
    first = candidate_answers[0]
    if isinstance(first, str):
        if len(question_list) != 1:
            raise ValueError("a flat candidate list requires exactly one question")
        groups = [[str(item) for item in candidate_answers]]
    else:
        groups = [[str(item) for item in group] for group in candidate_answers]
    if len(groups) != len(question_list):
        raise ValueError("candidate_answers must align one-to-one with questions")
    for question, group in zip(question_list, groups):
        if not question:
            raise ValueError("each question/prefix must be non-empty")
        if not group:
            raise ValueError("each question needs at least one candidate answer")
    return question_list, groups


def score_candidates(
    model,
    tokenizer,
    questions: str | Sequence[str],
    candidate_answers: Sequence[str] | Sequence[Sequence[str]],
    *,
    device: torch.device | None = None,
    accelerator=None,
) -> dict[str, Any]:
    """Unified candidate scorer used by listed InfoNCE.

    ``questions`` are already-rendered prefixes. For GRIP relation QA that means
    the chat string through the last ``<answer>`` tag, inclusive. Each candidate
    answer is the raw continuation string, normally a relation name. The scorer
    does not append ``</answer>`` or EOS.

    ``candidate_score`` is the length-normalized continuation log-probability
    of that answer under teacher forcing, identical to the tensor that listed
    training feeds into ``candidate_infonce_loss``. If a candidate tokenizes
    to ``L`` answer tokens:

        score(A | Q) = (1 / L) * sum_{t=1..L} log p(a_t | Q, a_<t)

    Prefix tokens, padding, EOS, and ``</answer>`` are excluded. Scoring uses
    the model's current dtype (bf16 during 7B training). Temperature is not
    applied here.

    Returns a dict with:

    - ``candidate_score``: per-question list of floats
    - ``candidate_token_length``: per-question list of answer token counts ``L``
    - ``scores`` / ``token_lengths``: packed tensors in the same row order
    """
    question_list, groups = _normalize_score_inputs(questions, candidate_answers)
    scorer = unwrap_for_scoring(model, accelerator)
    if device is None:
        device = next(scorer.parameters()).device
    rows: list[list[int]] = []
    prefix_lens: list[int] = []
    group_sizes: list[int] = []
    token_lengths: list[list[int]] = []
    for question, answers in zip(question_list, groups):
        packed_rows, packed_prefix = pack_decision_set_rows(tokenizer, question, answers)
        rows.extend(packed_rows)
        prefix_lens.extend(packed_prefix)
        group_sizes.append(len(answers))
        prefix_len = packed_prefix[0]
        token_lengths.append([len(row) - prefix_len for row in packed_rows])
    packed_scores, packed_lens = score_packed_candidate_rows(
        scorer,
        tokenizer,
        rows,
        prefix_lens,
        device,
        accelerator=accelerator,
    )
    offset = 0
    grouped_scores: list[list[float]] = []
    grouped_lens: list[list[int]] = []
    for size, expected_lens in zip(group_sizes, token_lengths):
        score_slice = packed_scores[offset : offset + size]
        length_slice = packed_lens[offset : offset + size]
        grouped_scores.append([float(value) for value in score_slice.detach().cpu()])
        observed = [int(value) for value in length_slice.detach().cpu()]
        if observed != expected_lens:
            raise ValueError(
                f"candidate token lengths from the packed rows {observed} "
                f"do not match tokenizer continuations {expected_lens}"
            )
        grouped_lens.append(observed)
        offset += size
    return {
        "candidate_score": grouped_scores,
        "candidate_token_length": grouped_lens,
        "scores": packed_scores,
        "token_lengths": packed_lens,
    }

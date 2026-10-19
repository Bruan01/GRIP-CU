"""Shared teacher-forced candidate scoring for InfoNCE and offline audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from constants import ANSWER_TAG, SYSTEM_PROMPT


@dataclass
class CandidateScoreResult:
    """Scores and valid answer-token counts for each question group."""

    scores: list[torch.Tensor]
    token_lengths: list[torch.Tensor]

    @property
    def candidate_score(self) -> list[torch.Tensor]:
        """Compatibility alias for the public score name."""
        return self.scores

    @property
    def candidate_token_length(self) -> list[torch.Tensor]:
        """Compatibility alias for the public token-length name."""
        return self.token_lengths


def _as_token_list(token_ids: Any) -> list[int]:
    """Normalize tokenizer output to one flat token-id list."""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu()
        if token_ids.ndim == 2:
            if token_ids.shape[0] != 1:
                raise ValueError("Expected one tokenized sequence, got a batch.")
            token_ids = token_ids[0]
        if token_ids.ndim != 1:
            raise ValueError("Expected 1D or single-row 2D token ids.")
        return [int(token_id) for token_id in token_ids.tolist()]
    if isinstance(token_ids, (list, tuple)):
        if len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple, torch.Tensor)):
            return _as_token_list(token_ids[0])
        return [int(token_id) for token_id in token_ids]
    return [int(token_ids)]


def _model_device(model) -> torch.device:
    """Get the model device for both Transformers models and light test doubles."""
    parameters = getattr(model, "parameters", None)
    if parameters is not None:
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    device = getattr(model, "device", None)
    if device is None:
        raise ValueError("Candidate scorer requires a model device.")
    return torch.device(device)


def _target_ids(tokenizer, candidates: Sequence[str]) -> list[list[int]]:
    return [
        _as_token_list(tokenizer(
            f"<{ANSWER_TAG}>{candidate}</{ANSWER_TAG}>",
            add_special_tokens=False,
        )["input_ids"])
        for candidate in candidates
    ]


def _prepare_record(
    tokenizer,
    record: dict[str, Any],
    max_length: int | None,
) -> tuple[list[int], list[list[int]]]:
    prompt_ids = record.get("_candidate_prompt_ids")
    target_id_rows = record.get("_candidate_target_ids")

    if prompt_ids is None:
        user_content = record.get(
            "user_content",
            f"Given the context graph titled {record['title']}, please answer the following question: "
            f"{record['question']} Response in the following format:"
        )
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": record.get("system_prompt", SYSTEM_PROMPT)},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = _as_token_list(tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=max_length is not None,
            **({"max_length": max_length} if max_length is not None else {}),
        )["input_ids"])
        record["_candidate_prompt_ids"] = prompt_ids

    if target_id_rows is None:
        target_id_rows = _target_ids(tokenizer, record["candidates"])
        record["_candidate_target_ids"] = target_id_rows

    normalized_targets = [_as_token_list(target_ids) for target_ids in target_id_rows]
    if any(not target_ids for target_ids in normalized_targets):
        raise ValueError("Candidate tokenization produced an empty answer sequence.")
    return _as_token_list(prompt_ids), normalized_targets


def score_candidate_records(
    model,
    tokenizer,
    records: Sequence[dict[str, Any]],
    *,
    max_length: int | None,
    score_clip: float | None = 20.0,
    candidate_batch_size: int | None = None,
) -> CandidateScoreResult:
    """Score every candidate continuation in each record.

    The score is the mean log probability of the candidate answer tokens only:
    ``sum(log p(answer_token)) / number_of_answer_tokens``. Prompt tokens,
    padding, and EOS are not included. The candidate text is wrapped as
    ``<answer>candidate</answer>`` exactly as in the training objective.
    """
    if not records:
        raise ValueError("At least one candidate record is required.")

    candidate_batch_size = candidate_batch_size or 0
    device = _model_device(model)
    padding_id = getattr(tokenizer, "pad_token_id", None)
    if padding_id is None:
        padding_id = getattr(tokenizer, "eos_token_id", None)
    if padding_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
    grouped_scores: list[torch.Tensor] = []
    grouped_lengths: list[torch.Tensor] = []

    for record in records:
        candidates = record.get("candidates")
        if not candidates:
            raise ValueError("Each candidate record must contain at least one candidate.")
        prompt_ids, target_id_rows = _prepare_record(tokenizer, record, max_length)
        prompt_rows: list[torch.Tensor] = []
        target_rows: list[torch.Tensor] = []
        max_target_length = max(len(target_ids) for target_ids in target_id_rows)
        if max_length is None:
            prompt_length = len(prompt_ids)
        else:
            prompt_length = min(len(prompt_ids), max(1, max_length - max_target_length))
        shared_prompt = torch.tensor(prompt_ids[:prompt_length], dtype=torch.long)
        for target_ids in target_id_rows:
            prompt_rows.append(shared_prompt)
            target_rows.append(torch.tensor(target_ids, dtype=torch.long))

        record_scores: list[torch.Tensor] = []
        record_lengths: list[int] = []
        step = candidate_batch_size if candidate_batch_size > 0 else len(target_rows)
        for start in range(0, len(target_rows), step):
            batch_prompts = prompt_rows[start:start + step]
            batch_targets = target_rows[start:start + step]
            max_target_length = max(target.numel() for target in batch_targets)
            input_ids = torch.full(
                (len(batch_targets), prompt_length + max_target_length),
                padding_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, (prompt, target) in enumerate(zip(batch_prompts, batch_targets)):
                target_length = target.numel()
                input_ids[row, :prompt_length] = prompt.to(device)
                input_ids[row, prompt_length:prompt_length + target_length] = target.to(device)
                attention_mask[row, :prompt_length + target_length] = 1

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=max_target_length + 1,
            ).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            for row, target in enumerate(batch_targets):
                target_length = target.numel()
                positions = log_probs[row, :target_length]
                score = positions.gather(
                    1, target.to(device).view(-1, 1)
                ).squeeze(1).mean()
                record_scores.append(score)
                record_lengths.append(target_length)

        scores = torch.stack(record_scores).float()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite candidate continuation score detected.")
        if score_clip is not None:
            scores = scores.clamp(min=-score_clip, max=score_clip)
        grouped_scores.append(scores)
        grouped_lengths.append(torch.tensor(record_lengths, dtype=torch.long, device=device))

    return CandidateScoreResult(grouped_scores, grouped_lengths)


def score_candidates(
    model,
    tokenizer,
    questions: Sequence[str | Mapping[str, Any]],
    candidate_answers: Sequence[Sequence[str]],
    *,
    max_length: int = 4096,
    score_clip: float | None = 20.0,
    candidate_batch_size: int | None = None,
) -> CandidateScoreResult:
    """Score one or more questions against their candidate answer lists."""
    if len(questions) != len(candidate_answers):
        raise ValueError("questions and candidate_answers must have the same length.")
    records: list[dict[str, Any]] = []
    for question, candidates in zip(questions, candidate_answers):
        record = dict(question) if isinstance(question, Mapping) else {"user_content": question}
        record["candidates"] = list(candidates)
        records.append(record)
    return score_candidate_records(
        model,
        tokenizer,
        records,
        max_length=max_length,
        score_clip=score_clip,
        candidate_batch_size=candidate_batch_size,
    )


def score_candidate_prompt(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    candidates: Sequence[str],
    *,
    score_clip: float | None = 20.0,
    candidate_batch_size: int | None = None,
) -> CandidateScoreResult:
    """Score candidates for an already-tokenized prompt without retruncating it."""
    record = {
        "_candidate_prompt_ids": prompt_ids[0].tolist() if prompt_ids.ndim == 2 else prompt_ids.tolist(),
        "candidates": list(candidates),
    }
    return score_candidate_records(
        model,
        tokenizer,
        [record],
        max_length=None,
        score_clip=score_clip,
        candidate_batch_size=candidate_batch_size,
    )

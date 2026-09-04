"""Prefix-constrained decoding over a per-query entity candidate set."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from transformers import LogitsProcessor


class CandidateTrieLogitsProcessor(LogitsProcessor):
    """Allow only token prefixes that occur in one candidate answer.

    ``generate`` calls a logits processor with the complete sequence.  The
    first ``start_length`` tokens are the prompt (including left padding); the
    remaining tokens are the answer continuation.  At a complete candidate we
    allow EOS, which makes exact candidate strings terminable even when one
    candidate is a prefix of another.

    This deliberately uses the two-argument ``__call__`` signature supported
    by current Transformers versions.  A previous implementation accepted an
    extra required argument and failed at runtime in Transformers 4.57.
    """

    def __init__(
        self,
        candidate_token_ids: list[list[list[int]]],
        start_length: int,
        eos_token_id: int,
        pad_token_id: int | None = None,
    ) -> None:
        super().__init__()
        if start_length < 0:
            raise ValueError("start_length must be non-negative")
        if not candidate_token_ids:
            raise ValueError("candidate_token_ids cannot be empty")
        if any(not candidates for candidates in candidate_token_ids):
            raise ValueError("each batch item needs at least one candidate")
        if any(not candidate for candidates in candidate_token_ids for candidate in candidates):
            raise ValueError("candidate answers must contain at least one token")
        self.candidate_token_ids = candidate_token_ids
        self.start_length = int(start_length)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id) if pad_token_id is not None else self.eos_token_id

    @staticmethod
    def _next_tokens(candidates: list[list[int]], prefix: list[int]) -> tuple[set[int], bool]:
        next_tokens: set[int] = set()
        complete = False
        for candidate in candidates:
            if prefix == candidate:
                complete = True
            elif len(prefix) < len(candidate) and candidate[: len(prefix)] == prefix:
                next_tokens.add(candidate[len(prefix)])
        return next_tokens, complete

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("input_ids and scores must be rank-2 tensors")
        if input_ids.shape[0] != len(self.candidate_token_ids):
            raise ValueError(
                f"batch mismatch: processor has {len(self.candidate_token_ids)} rows, "
                f"but generation passed {input_ids.shape[0]}"
            )
        if input_ids.shape[1] < self.start_length:
            raise ValueError("generation sequence is shorter than the prompt")
        constrained = torch.full_like(scores, torch.finfo(scores.dtype).min)
        for row_index, candidates in enumerate(self.candidate_token_ids):
            prefix = input_ids[row_index, self.start_length :].tolist()
            # Transformers may call processors for rows that already emitted
            # EOS while another row in the batch is still decoding.  Those
            # rows are subsequently padded by the generation loop.
            if self.eos_token_id in prefix:
                constrained[row_index, self.pad_token_id] = scores[row_index, self.pad_token_id]
                continue
            next_tokens, complete = self._next_tokens(candidates, prefix)
            if complete:
                next_tokens.add(self.eos_token_id)
            if not next_tokens:
                raise ValueError(
                    f"generated prefix is not in candidate trie for batch row {row_index}: {prefix}"
                )
            indices = torch.tensor(sorted(next_tokens), device=scores.device, dtype=torch.long)
            constrained[row_index, indices] = scores[row_index, indices]
        return constrained


def deduplicate_candidates(tokenizer, candidates: Iterable[str]) -> tuple[list[str], list[list[int]]]:
    """Return candidate strings and unique token sequences in stable order."""
    strings: list[str] = []
    token_ids: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        text = str(candidate)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        key = tuple(int(token) for token in ids)
        if not key or key in seen:
            continue
        seen.add(key)
        strings.append(text)
        token_ids.append(list(key))
    if not token_ids:
        raise ValueError("candidate list has no non-empty unique tokenization")
    return strings, token_ids

"""Trie-constrained greedy generation.

A :class:`CandidateTrieLogitsProcessor` forces ``model.generate`` to emit a
token sequence that is *exactly* one of the formatted candidate answers
``<answer>{candidate}</answer>``. At every step only trie-children tokens keep
non-masked logits; once a terminal candidate is complete the processor forces
``eos`` so generation stops on a valid candidate.
"""

from __future__ import annotations

import math
import torch

_NEG_INF = -math.inf


class _TrieNode:
    __slots__ = ("children", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.is_terminal: bool = False


def build_candidate_trie(tokenizer, candidates: list[str]) -> _TrieNode:
    """Build a prefix trie over ``<answer>{cand}</answer>`` token sequences."""

    root = _TrieNode()
    for candidate in candidates:
        token_ids = tokenizer.encode(
            f"<answer>{candidate}</answer>", add_special_tokens=False
        )
        node = root
        for tok in token_ids:
            child = node.children.get(tok)
            if child is None:
                child = _TrieNode()
                node.children[tok] = child
            node = child
        node.is_terminal = True
    return root


def _walk_trie(root: _TrieNode, generated_ids: list[int]) -> _TrieNode:
    """Return the trie node reached by following *generated_ids* from *root*.

    If the generated prefix is not on any candidate path the walk stops at the
    deepest valid node; the caller treats a missing-children state as
    "force eos".
    """

    node = root
    for tok in generated_ids:
        child = node.children.get(tok)
        if child is None:
            return node  # off-path: no valid continuation
        node = child
    return node


class CandidateTrieLogitsProcessor:
    """HuggingFace-compatible logits processor (batch size 1).

    The processor is stateless across ``generate`` calls except for the trie
    and the prompt length, which are fixed at construction. Per-step state is
    re-derived from the full ``input_ids`` passed to ``__call__``.
    """

    def __init__(
        self,
        trie: _TrieNode,
        prompt_length: int,
        eos_token_id: int,
        vocab_size: int,
    ) -> None:
        if prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        self.trie = trie
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if input_ids.shape[0] != 1:
            raise ValueError("CandidateTrieLogitsProcessor supports batch size 1 only")
        generated = input_ids[0, self.prompt_length :].tolist()
        node = _walk_trie(self.trie, generated)

        masked = scores.clone()
        # No continuation is valid from this node -> only eos is allowed.
        if not node.children:
            allowed = {self.eos_token_id} if node.is_terminal else set()
        else:
            allowed = set(node.children.keys())
            if node.is_terminal:
                allowed.add(self.eos_token_id)

        if not allowed:
            # Completely off-path with no eos fallback: force eos to terminate
            # gracefully rather than emitting arbitrary tokens.
            allowed = {self.eos_token_id}

        mask = torch.full((self.vocab_size,), _NEG_INF, device=scores.device)
        for tok in allowed:
            mask[tok] = 0.0
        return masked + mask.unsqueeze(0)

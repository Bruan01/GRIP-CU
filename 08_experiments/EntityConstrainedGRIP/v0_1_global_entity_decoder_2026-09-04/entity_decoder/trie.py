"""Token trie and Hugging Face-compatible constrained-decoding callback."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass
class _Node:
    children: dict[int, "_Node"] = field(default_factory=dict)
    entity: str | None = None


class EntityTokenTrie:
    def __init__(self, entity_tokens: Mapping[str, Sequence[int]]) -> None:
        if not entity_tokens:
            raise ValueError("entity token mapping is empty")
        self.root = _Node()
        self._sequences: dict[tuple[int, ...], str] = {}
        for entity, raw_tokens in entity_tokens.items():
            tokens = tuple(int(token) for token in raw_tokens)
            if not tokens:
                raise ValueError(f"empty tokenization for entity {entity!r}")
            if tokens in self._sequences and self._sequences[tokens] != entity:
                raise ValueError(f"tokenization collision: {self._sequences[tokens]!r} and {entity!r}")
            self._sequences[tokens] = str(entity)
            node = self.root
            for token in tokens:
                node = node.children.setdefault(token, _Node())
            node.entity = str(entity)

    @classmethod
    def from_tokenizer(cls, entities: Iterable[str], tokenizer) -> "EntityTokenTrie":
        mapping = {}
        for entity in entities:
            tokens = tokenizer(str(entity), add_special_tokens=False)["input_ids"]
            mapping[str(entity)] = tokens
        return cls(mapping)

    def _node_for(self, prefix: Sequence[int]) -> _Node | None:
        node = self.root
        for token in prefix:
            node = node.children.get(int(token))
            if node is None:
                return None
        return node

    def allowed_next(self, prefix: Sequence[int]) -> set[int]:
        node = self._node_for(prefix)
        return set(node.children) if node is not None else set()

    def is_terminal(self, tokens: Sequence[int]) -> bool:
        node = self._node_for(tokens)
        return bool(node is not None and node.entity is not None)

    def entity_for(self, tokens: Sequence[int]) -> str:
        node = self._node_for(tokens)
        if node is None or node.entity is None:
            raise KeyError(tuple(tokens))
        return node.entity

    @property
    def token_sequences(self) -> Mapping[tuple[int, ...], str]:
        return dict(self._sequences)


class TrieConstraint:
    """Callable seam used as ``prefix_allowed_tokens_fn`` by ``generate``."""

    def __init__(self, trie: EntityTokenTrie, prompt_length: int | Sequence[int], eos_token_id: int) -> None:
        self.trie = trie
        self.prompt_lengths = ([int(prompt_length)] if isinstance(prompt_length, int) else [int(v) for v in prompt_length])
        self.eos_token_id = int(eos_token_id)

    def allowed_tokens(self, batch_id: int, input_ids: Sequence[int]) -> list[int]:
        prompt_length = self.prompt_lengths[0] if len(self.prompt_lengths) == 1 else self.prompt_lengths[int(batch_id)]
        generated = [int(value) for value in input_ids[prompt_length:]]
        allowed = sorted(self.trie.allowed_next(generated))
        if self.trie.is_terminal(generated):
            return sorted(set(allowed + [self.eos_token_id]))
        return allowed or [self.eos_token_id]

    def __call__(self, batch_id: int, input_ids) -> list[int]:
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return self.allowed_tokens(batch_id, input_ids)

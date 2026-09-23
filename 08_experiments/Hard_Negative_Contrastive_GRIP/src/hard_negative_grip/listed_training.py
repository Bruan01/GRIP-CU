"""Listed-decision-set contrastive trainer pieces.

These wrap Hugging Face ``Trainer`` without changing the RecurrentGRIP snapshot.
Stage-1 context training stays the original generation loss. Stage-2 QA adds
InfoNCE over the prompt's 10-way distractors when ``lambda_candidate > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling, Trainer

from .losses import candidate_infonce_loss
from .official_lists import listed_relations_from_sample
from .scoring import (  # re-exported for tests and eval scripts
    encode_without_specials,
    pack_decision_set_rows,
    score_candidate_rows,
    unwrap_for_scoring,
)


EXTRA_KEYS = (
    "listed_relations",
    "positive_relation",
    "prefix_text",
    "prefix_ids",
    "relation_ids",
)


def format_answer_prefix(
    tokenizer,
    *,
    title: str,
    question: str,
    system_prompt: str,
    question_template: str,
) -> str:
    """Chat prefix ending immediately after the literal ``<answer>`` tag."""
    user_content = question_template.format(title=title, question=question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return base + "<answer>"


def listed_negatives(sample: dict) -> list[str]:
    """Non-gold relations from the question's own 10-way list."""
    positive = str(sample.get("answer", ""))
    return [rel for rel in listed_relations_from_sample(sample) if rel and rel != positive]


@dataclass
class ListedDataCollator:
    tokenizer: Any

    def __post_init__(self) -> None:
        self.inner = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def __call__(self, features: list[dict]) -> dict:
        extra = {key: [feature.pop(key, None) for feature in features] for key in EXTRA_KEYS}
        batch = self.inner(features)
        batch.update(extra)
        return batch


def pick_closed_set_answer(relations: list[str], scores: list[float]) -> str:
    """Argmax over listed continuations; earlier prompt order wins ties."""
    if not relations:
        raise ValueError("relations must be non-empty")
    if len(relations) != len(scores):
        raise ValueError("relations and scores must have the same length")
    best_index = 0
    best_score = scores[0]
    for index, score in enumerate(scores):
        if score > best_score:
            best_index = index
            best_score = score
    return relations[best_index]


class ListedQADataset(torch.utils.data.Dataset):
    """Tokenized QA strings plus the 10-way list used at evaluation."""

    def __init__(
        self,
        texts: list[str],
        metas: list[dict],
        tokenizer,
    ) -> None:
        if len(texts) != len(metas):
            raise ValueError("texts and metas must have the same length")
        self.tokenizer = tokenizer
        self.items = [
            self._encode_item(text, meta, tokenizer) for text, meta in zip(texts, metas)
        ]

    @staticmethod
    def _encode_item(text: str, meta: dict, tokenizer) -> dict:
        encoded = tokenizer(text, truncation=True, padding=False)
        prefix = meta.get("prefix_text") or ""
        positive = meta.get("positive_relation") or ""
        listed = [rel for rel in (meta.get("listed_relations") or []) if rel and rel != positive]
        prefix_ids: list[int] = []
        relation_ids: list[list[int]] = []
        if prefix and positive:
            prefix_ids = encode_without_specials(tokenizer, prefix)
            relation_ids = [encode_without_specials(tokenizer, positive)]
            relation_ids.extend(encode_without_specials(tokenizer, rel) for rel in listed)
        return {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "listed_relations": list(meta.get("listed_relations") or []),
            "positive_relation": positive,
            "prefix_text": prefix,
            "prefix_ids": prefix_ids,
            "relation_ids": relation_ids,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        return {
            "input_ids": list(item["input_ids"]),
            "attention_mask": list(item["attention_mask"]),
            "listed_relations": list(item["listed_relations"]),
            "positive_relation": item["positive_relation"],
            "prefix_text": item["prefix_text"],
            "prefix_ids": list(item["prefix_ids"]),
            "relation_ids": [list(row) for row in item["relation_ids"]],
        }


class ListedContrastiveTrainer(Trainer):
    """Generation loss plus optional listed-candidate InfoNCE."""

    def __init__(
        self,
        *args,
        lambda_candidate: float = 0.0,
        temperature: float = 1.0,
        memory_size: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if lambda_candidate < 0:
            raise ValueError("lambda_candidate must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if memory_size < 0:
            raise ValueError("memory_size must be non-negative")
        self.lambda_candidate = lambda_candidate
        self.temperature = temperature
        self.memory_size = memory_size
        self.relation_memory: list[list[int]] = []
        self.candidate_forwards = 0
        self.memory_forwards = 0
        self.last_generation_loss = 0.0
        self.last_candidate_loss = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        listed_batch = inputs.pop("listed_relations", None)
        positive_batch = inputs.pop("positive_relation", None)
        prefix_batch = inputs.pop("prefix_text", None)
        prefix_ids_batch = inputs.pop("prefix_ids", None)
        relation_ids_batch = inputs.pop("relation_ids", None)
        outputs = model(**inputs)
        generation_loss = outputs.loss
        candidate_loss = generation_loss.new_zeros(())
        if self.lambda_candidate > 0:
            candidate_loss = self._listed_candidate_loss(
                model,
                prefix_ids_batch or [],
                relation_ids_batch or [],
                prefixes=prefix_batch or [],
                positives=positive_batch or [],
                listed_lists=listed_batch or [],
            )
        self.last_generation_loss = float(generation_loss.detach())
        self.last_candidate_loss = float(candidate_loss.detach())
        loss = generation_loss + self.lambda_candidate * candidate_loss
        return (loss, outputs) if return_outputs else loss

    def _listed_candidate_loss(
        self,
        model,
        prefix_ids_batch,
        relation_ids_batch,
        *,
        prefixes,
        positives,
        listed_lists,
    ) -> torch.Tensor:
        tokenizer = self.processing_class
        scorer = unwrap_for_scoring(model, getattr(self, "accelerator", None))
        device = next(scorer.parameters()).device
        rows: list[list[int]] = []
        prefix_lens: list[int] = []
        group_sizes: list[int] = []
        for prefix_ids, relation_ids, prefix, positive, listed in zip(
            prefix_ids_batch,
            relation_ids_batch,
            prefixes,
            positives,
            listed_lists,
        ):
            packed_ids, packed_prefix = self._candidate_token_rows(
                tokenizer,
                prefix_ids=prefix_ids,
                relation_ids=relation_ids,
                prefix=prefix,
                positive=positive,
                listed=listed,
                memory=self.relation_memory,
                memory_size=self.memory_size,
            )
            if packed_ids is None:
                group_sizes.append(0)
                continue
            rows.extend(packed_ids)
            prefix_lens.extend([packed_prefix] * len(packed_ids))
            group_sizes.append(len(packed_ids))
        if not rows:
            return next(scorer.parameters()).new_zeros(())
        scores = self._score_candidate_rows(scorer, tokenizer, rows, prefix_lens, device)
        losses = []
        offset = 0
        for size in group_sizes:
            if size < 2:
                offset += size
                continue
            group = scores[offset : offset + size]
            offset += size
            losses.append(
                candidate_infonce_loss(
                    group[:1],
                    group[1:].unsqueeze(0),
                    temperature=self.temperature,
                )
            )
        if self.memory_size > 0:
            fresh = []
            for relation_ids in relation_ids_batch:
                for row in relation_ids[1:]:
                    if row and tuple(row) not in {tuple(item) for item in self.relation_memory + fresh}:
                        fresh.append(list(row))
            self.relation_memory = (self.relation_memory + fresh)[-self.memory_size:]
        if not losses:
            return next(scorer.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def _candidate_token_rows(
        self,
        tokenizer,
        *,
        prefix_ids,
        relation_ids,
        prefix,
        positive,
        listed,
        memory,
        memory_size,
    ) -> tuple[list[list[int]] | None, int]:
        ids = [list(row) for row in (relation_ids or []) if row]
        pids = list(prefix_ids or [])
        if (not pids or len(ids) < 2) and prefix and positive:
            pids = encode_without_specials(tokenizer, prefix)
            negatives = [rel for rel in (listed or []) if rel and rel != positive]
            if negatives:
                ids = [encode_without_specials(tokenizer, positive)]
                ids.extend(encode_without_specials(tokenizer, rel) for rel in negatives)
        if memory_size > 0 and memory:
            seen = {tuple(row) for row in ids}
            for row in memory:
                if tuple(row) not in seen:
                    ids.append(list(row))
                    seen.add(tuple(row))
                if len(ids) >= 2 + len(memory):
                    break
        if not pids or len(ids) < 2:
            return None, 0
        return [pids + continuation for continuation in ids], len(pids)

    def _score_candidate_rows(self, scorer, tokenizer, rows, prefix_lens, device) -> torch.Tensor:
        self.candidate_forwards += len(rows)
        return score_candidate_rows(
            scorer,
            tokenizer,
            rows,
            prefix_lens,
            device,
            accelerator=getattr(self, "accelerator", None),
        )

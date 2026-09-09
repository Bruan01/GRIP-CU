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
from .scoring import normalized_continuation_log_likelihood


EXTRA_KEYS = ("listed_relations", "positive_relation", "prefix_text")


def unwrap_for_scoring(model, accelerator=None):
    """Avoid Accelerate wrapping, which materializes full fp32 logits."""
    if accelerator is not None:
        return accelerator.unwrap_model(model)
    return model


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
        self.texts = texts
        self.metas = metas
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(self.texts[index], truncation=True, padding=False)
        meta = self.metas[index]
        encoded["listed_relations"] = list(meta.get("listed_relations") or [])
        encoded["positive_relation"] = meta.get("positive_relation") or ""
        encoded["prefix_text"] = meta.get("prefix_text") or ""
        return encoded


class ListedContrastiveTrainer(Trainer):
    """Generation loss plus optional listed-candidate InfoNCE."""

    def __init__(
        self,
        *args,
        lambda_candidate: float = 0.0,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if lambda_candidate < 0:
            raise ValueError("lambda_candidate must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.lambda_candidate = lambda_candidate
        self.temperature = temperature
        self.candidate_forwards = 0
        self.last_generation_loss = 0.0
        self.last_candidate_loss = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        listed_batch = inputs.pop("listed_relations", None)
        positive_batch = inputs.pop("positive_relation", None)
        prefix_batch = inputs.pop("prefix_text", None)
        outputs = model(**inputs)
        generation_loss = outputs.loss
        candidate_loss = generation_loss.new_zeros(())
        if self.lambda_candidate > 0:
            candidate_loss = self._listed_candidate_loss(
                model,
                prefix_batch or [],
                positive_batch or [],
                listed_batch or [],
            )
        self.last_generation_loss = float(generation_loss.detach())
        self.last_candidate_loss = float(candidate_loss.detach())
        loss = generation_loss + self.lambda_candidate * candidate_loss
        return (loss, outputs) if return_outputs else loss

    def _listed_candidate_loss(self, model, prefixes, positives, listed_lists) -> torch.Tensor:
        tokenizer = self.processing_class
        scorer = unwrap_for_scoring(model, getattr(self, "accelerator", None))
        device = next(scorer.parameters()).device
        losses = []
        for prefix, positive, listed in zip(prefixes, positives, listed_lists):
            if not prefix or not positive:
                continue
            negatives = [rel for rel in (listed or []) if rel and rel != positive]
            if not negatives:
                continue
            relations = [positive] + negatives
            prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
            scores = []
            for relation in relations:
                continuation = tokenizer(relation, add_special_tokens=False).input_ids
                if not continuation:
                    continuation = [tokenizer.unk_token_id or 0]
                row = prefix_ids + continuation
                input_ids = torch.tensor([row], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                outputs = scorer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                self.candidate_forwards += 1
                score = normalized_continuation_log_likelihood(
                    outputs.logits,
                    input_ids,
                    torch.tensor([len(prefix_ids)], dtype=torch.long, device=device),
                    torch.tensor([len(row)], dtype=torch.long, device=device),
                )
                scores.append(score.float().reshape(()))
                del outputs
            stacked = torch.stack(scores)
            losses.append(
                candidate_infonce_loss(
                    stacked[:1],
                    stacked[1:].unsqueeze(0),
                    temperature=self.temperature,
                )
            )
        if not losses:
            return next(scorer.parameters()).new_zeros(())
        return torch.stack(losses).mean()

"""Tokenization, deterministic token budgets, and causal-LM collation."""

from __future__ import annotations

import random
from typing import Sequence


def tokenize_supervision_rows(rows: Sequence[dict], tokenizer, max_length: int) -> list[dict]:
    tokenized: list[dict] = []
    eos = tokenizer.eos_token or ""
    for row in rows:
        raw_prompt_ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
        target_text = str(row.get("target_text", row["answer"]))
        answer_ids = tokenizer(target_text + eos, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= max_length:
            raise ValueError(f"answer exceeds max_length for {row['task_id']}")
        prompt_budget = max_length - len(answer_ids)
        truncated_prompt_tokens = max(0, len(raw_prompt_ids) - prompt_budget)
        prompt_ids = raw_prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + answer_ids
        tokenized.append(
            {
                **row,
                "input_ids": input_ids,
                "labels": [-100] * len(prompt_ids) + answer_ids,
                "input_token_count": len(input_ids),
                "supervised_token_count": len(answer_ids),
                "target_text": target_text,
                "raw_prompt_token_count": len(raw_prompt_ids),
                "truncated_prompt_tokens": truncated_prompt_tokens,
            }
        )
    return tokenized


def balance_rows_to_token_budget(rows: Sequence[dict], target_input_tokens: int, seed: int) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("cannot balance empty rows")
    if target_input_tokens <= 0:
        raise ValueError("target_input_tokens must be positive")
    selected: list[dict] = []
    total = 0
    cycle = 0
    canonical = sorted(rows, key=lambda row: row["task_id"])
    while total < target_input_tokens:
        order = list(canonical)
        random.Random(seed + cycle * 1000003).shuffle(order)
        for row in order:
            selected.append(row)
            total += int(row["input_token_count"])
            if total >= target_input_tokens:
                break
        cycle += 1
        if cycle > 10000:
            raise RuntimeError("token-budget balancing exceeded safety limit")
    return selected, {
        "target_input_tokens": int(target_input_tokens),
        "input_tokens": total,
        "overshoot_tokens": total - int(target_input_tokens),
        "examples": len(selected),
        "cycles": cycle,
        "supervised_tokens": sum(int(row.get("supervised_token_count", 0)) for row in selected),
        "truncated_examples": sum(int(row.get("truncated_prompt_tokens", 0)) > 0 for row in selected),
        "truncated_prompt_tokens": sum(int(row.get("truncated_prompt_tokens", 0)) for row in selected),
    }


class TokenizedDataset:
    def __init__(self, rows: Sequence[dict]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: Sequence[dict]) -> dict:
        import torch

        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        labels = []
        attention_mask = []
        for example in examples:
            pad = max_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * pad)
            labels.append(example["labels"] + [-100] * pad)
            attention_mask.append([1] * len(example["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "task_ids": [example["task_id"] for example in examples],
            "depths": torch.tensor([example["depth_label"] for example in examples], dtype=torch.long),
            "input_token_count": sum(example["input_token_count"] for example in examples),
            "supervised_token_count": sum(example["supervised_token_count"] for example in examples),
        }

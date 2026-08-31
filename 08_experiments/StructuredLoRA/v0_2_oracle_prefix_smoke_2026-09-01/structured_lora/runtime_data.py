"""Torch dataset and collation for exact-hop causal language modeling."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from .records import build_prompt


class ExactHopDataset(Dataset):
    def __init__(self, rows: Sequence[dict], tokenizer, max_length: int) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prompt = build_prompt(row["text"])
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_text = str(row["answer"]) + (self.tokenizer.eos_token or "")
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= self.max_length:
            raise ValueError(f"answer exceeds max_length for {row['task_id']}")
        prompt_budget = self.max_length - len(answer_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": input_ids,
            "labels": labels,
            "depth": int(row["depth_label"]),
            "task_id": row["task_id"],
            "answer": row["answer"],
        }


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: Sequence[dict]) -> dict:
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
            "depths": torch.tensor([example["depth"] for example in examples], dtype=torch.long),
            "task_ids": [example["task_id"] for example in examples],
            "answers": [example["answer"] for example in examples],
        }

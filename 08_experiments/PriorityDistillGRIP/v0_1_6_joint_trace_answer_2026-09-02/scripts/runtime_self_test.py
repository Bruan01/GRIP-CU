#!/usr/bin/env python3
"""Fast CPU check for independent LoRA injection and adapter serialization."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
from torch import nn

from priority_distill.modules import adapter_state_dict, inject_lora, load_adapter_state_dict


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(8, 6, bias=False)
        self.up_proj = nn.Linear(6, 8, bias=False)
        self.gate_proj = nn.Linear(8, 6, bias=False)

    def forward(self, x):
        return self.up_proj(torch.relu(self.down_proj(x)) * torch.sigmoid(self.gate_proj(x)))


def main() -> None:
    torch.manual_seed(7)
    model = Toy()
    report = inject_lora(model, target_modules=["down_proj", "up_proj", "gate_proj"], rank=2, alpha=4, dropout=0.0, orthogonal_init=False)
    output = model(torch.randn(3, 8)).sum()
    output.backward()
    state = adapter_state_dict(model)
    if report.trainable_parameters <= 0 or not state:
        raise AssertionError("LoRA injection produced no trainable state")
    clone = Toy()
    inject_lora(clone, target_modules=["down_proj", "up_proj", "gate_proj"], rank=2, alpha=4, dropout=0.0, orthogonal_init=False)
    load_adapter_state_dict(clone, state)
    print(f"runtime self-test passed: replaced={len(report.replaced_modules)} trainable={report.trainable_parameters}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Torch-level self-test for routing gradients and equal parameter counts."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
from torch import nn

from structured_lora.modules import MonolithicLoraLinear, RoutingController, StructuredLoraLinear


def nonzero_group_gradients(module: StructuredLoraLinear) -> list[bool]:
    grad = module.lora_B.grad
    if grad is None:
        return [False] * module.groups
    return [bool(torch.count_nonzero(grad[group]).item()) for group in range(module.groups)]


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_a = nn.Linear(16, 12, bias=False, device=device, dtype=torch.float32)
    base_b = nn.Linear(16, 12, bias=False, device=device, dtype=torch.float32)
    base_b.weight.data.copy_(base_a.weight.data)
    mono_controller = RoutingController("monolithic", groups=4)
    mono = MonolithicLoraLinear(
        base_a,
        rank=8,
        alpha=16,
        dropout=0,
        groups=4,
        controller=mono_controller,
        orthogonal_init=True,
    )
    local_controller = RoutingController("ordered_prefix_local_credit", groups=4)
    local = StructuredLoraLinear(
        base_b,
        groups=4,
        group_rank=2,
        total_rank=8,
        alpha=16,
        dropout=0,
        controller=local_controller,
        orthogonal_init=True,
    )
    mono_params = sum(parameter.numel() for parameter in mono.parameters() if parameter.requires_grad)
    local_params = sum(parameter.numel() for parameter in local.parameters() if parameter.requires_grad)
    if mono_params != local_params:
        raise AssertionError(f"parameter mismatch: monolithic={mono_params}, structured={local_params}")

    x = torch.randn(2, 3, 16, device=device)
    local_controller.set_batch([3, 3])
    local(x).sum().backward()
    active = nonzero_group_gradients(local)
    if active != [False, False, True, False]:
        raise AssertionError(f"depth-local credit leaked gradients: {active}")

    local.zero_grad(set_to_none=True)
    local_controller.method = "ordered_prefix"
    local_controller.set_batch([3, 3])
    local(x).sum().backward()
    active = nonzero_group_gradients(local)
    if active != [True, True, True, False]:
        raise AssertionError(f"prefix gradients do not match forward prefix: {active}")

    print(
        f"runtime self-test PASS device={device} equal_trainable_params={mono_params} "
        "local_credit=[0,0,1,0] prefix=[1,1,1,0]"
    )


if __name__ == "__main__":
    main()

"""Independent snapshot of the standard equal-rank LoRA used in v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class InjectionReport:
    target_modules: tuple[str, ...]
    replaced_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def _orthogonal_rows(rows: int, columns: int, device: torch.device) -> torch.Tensor:
    if rows > columns:
        raise ValueError(f"cannot initialize {rows} orthogonal rows in {columns} dimensions")
    matrix = torch.randn(columns, rows, dtype=torch.float32, device=device)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.transpose(0, 1).contiguous()


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float, orthogonal_init: bool) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        with torch.no_grad():
            if orthogonal_init:
                self.lora_A.copy_(_orthogonal_rows(rank, base.in_features, base.weight.device))
            else:
                nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        update = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base_output + update.to(base_output.dtype) * self.scaling


def _resolve_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
    orthogonal_init: bool,
) -> InjectionReport:
    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = tuple(target_modules)
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets
    ]
    if not replacements:
        raise ValueError(f"no nn.Linear modules matched {targets}")
    replaced_names = []
    for name, base in replacements:
        parent, child_name = _resolve_parent(model, name)
        setattr(parent, child_name, LoraLinear(base, rank, alpha, dropout, orthogonal_init))
        replaced_names.append(name)
    return InjectionReport(
        target_modules=targets,
        replaced_modules=tuple(replaced_names),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (name.endswith("lora_A") or name.endswith("lora_B"))
    }


def load_adapter_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    unknown = sorted(set(state) - set(named))
    if unknown:
        raise KeyError(f"adapter contains unknown parameters: {unknown[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))

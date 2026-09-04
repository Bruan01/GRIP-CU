"""Independent copy of the equal-rank LoRA architecture used by Phase-A checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class InjectionReport:
    target_modules: tuple[str, ...]
    replaced_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def inject_lora(model, *, target_modules: Iterable[str], rank: int, alpha: float, dropout: float = 0.0, orthogonal_init: bool = False) -> InjectionReport:
    import torch
    from torch import nn
    from torch.nn import functional as F

    class LoraLinear(nn.Module):
        def __init__(self, base: nn.Linear) -> None:
            super().__init__()
            self.base = base
            self.scaling = float(alpha) / float(rank)
            self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
            self.lora_A = nn.Parameter(torch.empty(int(rank), base.in_features, dtype=torch.float32, device=base.weight.device))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, int(rank), dtype=torch.float32, device=base.weight.device))
            with torch.no_grad():
                if orthogonal_init:
                    if int(rank) > base.in_features:
                        raise ValueError("rank exceeds input dimension for orthogonal initialization")
                    matrix = torch.randn(base.in_features, int(rank), dtype=torch.float32, device=base.weight.device)
                    q, _ = torch.linalg.qr(matrix, mode="reduced")
                    self.lora_A.copy_(q.transpose(0, 1).contiguous())
                else:
                    nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            for parameter in self.base.parameters():
                parameter.requires_grad = False

        def forward(self, x):
            base_output = self.base(x)
            update = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
            return base_output + update.to(base_output.dtype) * self.scaling

    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = tuple(target_modules)
    replacements = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets]
    if not replacements:
        raise ValueError(f"no nn.Linear modules matched {targets}")
    names = []
    for name, base in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoraLinear(base))
        names.append(name)
    return InjectionReport(
        target_modules=targets,
        replaced_modules=tuple(names),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def load_adapter_state_dict(model, state: Mapping) -> None:
    named = dict(model.named_parameters())
    unknown = sorted(set(state) - set(named))
    missing = sorted(name for name in named if (name.endswith("lora_A") or name.endswith("lora_B")) and name not in state)
    if unknown or missing:
        raise KeyError(f"adapter key mismatch: unknown={unknown[:5]} missing={missing[:5]}")
    with __import__("torch").no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))

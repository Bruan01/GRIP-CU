"""Torch modules for equal-rank monolithic and depth-structured LoRA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .routing import build_route_masks


@dataclass
class InjectionReport:
    method: str
    target_modules: tuple[str, ...]
    replaced_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


class RoutingController:
    """Shared per-batch routing state used by every injected linear layer."""

    def __init__(
        self,
        method: str,
        *,
        groups: int,
        depth_permutation: Mapping[int, int] | None = None,
        group_order: Sequence[int] | None = None,
        non_nested_masks: Mapping[int, Sequence[int]] | None = None,
    ) -> None:
        self.method = method
        self.groups = groups
        self.depth_permutation = dict(depth_permutation or {})
        self.group_order = tuple(group_order) if group_order is not None else None
        self.non_nested_masks = dict(non_nested_masks or {})
        self.depths: tuple[int, ...] | None = None
        self.knockout_group: int | None = None
        self._cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def set_batch(self, depths: Iterable[int]) -> None:
        self.depths = tuple(int(depth) for depth in depths)
        self._cache.clear()

    def set_knockout(self, group: int | None) -> None:
        self.knockout_group = group
        self._cache.clear()

    def gates(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.depths is None:
            raise RuntimeError("routing depths were not set before model forward")
        key = (str(device), str(dtype))
        if key not in self._cache:
            masks = build_route_masks(
                self.method,
                self.depths,
                groups=self.groups,
                depth_permutation=self.depth_permutation or None,
                group_order=self.group_order,
                non_nested_masks=self.non_nested_masks or None,
                knockout_group=self.knockout_group,
            )
            self._cache[key] = (
                torch.tensor(masks.forward, device=device, dtype=dtype),
                torch.tensor(masks.credit, device=device, dtype=dtype),
            )
        return self._cache[key]


def _broadcast_gate(gate: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    if gate.shape[0] != update.shape[0]:
        if gate.shape[0] == 1:
            gate = gate.expand(update.shape[0])
        else:
            raise RuntimeError(
                f"routing batch {gate.shape[0]} does not match activation batch {update.shape[0]}"
            )
    return gate.reshape(gate.shape[0], *([1] * (update.ndim - 1)))


def _orthogonal_rows(rows: int, columns: int, *, device: torch.device | None = None) -> torch.Tensor:
    if rows > columns:
        raise ValueError(f"cannot initialize {rows} orthogonal rows in dimension {columns}")
    matrix = torch.randn(columns, rows, dtype=torch.float32, device=device)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.transpose(0, 1).contiguous()


class MonolithicLoraLinear(nn.Module):
    """Standard rank-r LoRA with rank-slice knockout only for analysis."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        groups: int,
        controller: RoutingController,
        orthogonal_init: bool,
    ) -> None:
        super().__init__()
        if rank % groups:
            raise ValueError("monolithic rank must be divisible by groups for matched knockout slices")
        self.base = base
        self.rank = rank
        self.groups = groups
        self.group_rank = rank // groups
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.controller = controller
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        with torch.no_grad():
            if orthogonal_init:
                self.lora_A.copy_(_orthogonal_rows(rank, base.in_features, device=base.weight.device))
            else:
                nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        lora_input = self.dropout(x).to(self.lora_A.dtype)
        low_rank = F.linear(lora_input, self.lora_A)
        if self.controller.knockout_group is not None:
            mask = torch.ones(self.rank, device=low_rank.device, dtype=low_rank.dtype)
            start = self.controller.knockout_group * self.group_rank
            mask[start : start + self.group_rank] = 0
            low_rank = low_rank * mask
        update = F.linear(low_rank, self.lora_B)
        return base_output + update.to(base_output.dtype) * self.scaling


class StructuredLoraLinear(nn.Module):
    """Group-structured LoRA with separate forward and gradient-credit masks."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        groups: int,
        group_rank: int,
        total_rank: int,
        alpha: float,
        dropout: float,
        controller: RoutingController,
        orthogonal_init: bool,
    ) -> None:
        super().__init__()
        if groups * group_rank != total_rank:
            raise ValueError("groups * group_rank must equal total_rank")
        self.base = base
        self.groups = groups
        self.group_rank = group_rank
        self.total_rank = total_rank
        # Use the total-rank denominator for every group. All-on static split is
        # therefore scale-matched to monolithic rank-r LoRA.
        self.scaling = float(alpha) / float(total_rank)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.controller = controller
        self.lora_A = nn.Parameter(
            torch.empty(groups, group_rank, base.in_features, dtype=torch.float32, device=base.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(groups, base.out_features, group_rank, dtype=torch.float32, device=base.weight.device)
        )
        with torch.no_grad():
            if orthogonal_init:
                rows = _orthogonal_rows(total_rank, base.in_features, device=base.weight.device)
                self.lora_A.copy_(rows.reshape(groups, group_rank, base.in_features))
            else:
                for group in range(groups):
                    nn.init.kaiming_uniform_(self.lora_A[group], a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        lora_input = self.dropout(x).to(self.lora_A.dtype)
        forward_gates, credit_gates = self.controller.gates(lora_input.device, lora_input.dtype)
        combined = torch.zeros_like(base_output, dtype=self.lora_A.dtype)
        for group in range(self.groups):
            update = F.linear(F.linear(lora_input, self.lora_A[group]), self.lora_B[group])
            forward_gate = _broadcast_gate(forward_gates[:, group], update)
            credit_gate = _broadcast_gate(credit_gates[:, group], update)
            # Forward value equals forward_gate * update. The gradient enters
            # only through credit_gate, which can be one-hot for local credit.
            routed = update * credit_gate + update.detach() * (forward_gate - credit_gate)
            combined = combined + routed
        return base_output + combined.to(base_output.dtype) * self.scaling


def _resolve_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    method: str,
    target_modules: Sequence[str],
    total_rank: int,
    groups: int,
    group_rank: int,
    alpha: float,
    dropout: float,
    orthogonal_init: bool,
    depth_permutation: Mapping[int, int] | None = None,
    group_order: Sequence[int] | None = None,
    non_nested_masks: Mapping[int, Sequence[int]] | None = None,
) -> tuple[RoutingController, InjectionReport]:
    """Freeze the base model and replace matching linear layers in-place."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    controller = RoutingController(
        method,
        groups=groups,
        depth_permutation=depth_permutation,
        group_order=group_order,
        non_nested_masks=non_nested_masks,
    )
    targets = tuple(target_modules)
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            replacements.append((name, module))
    if not replacements:
        raise ValueError(f"no nn.Linear modules matched target names {targets}")

    replaced_names: list[str] = []
    for name, base in replacements:
        parent, child_name = _resolve_parent(model, name)
        if method == "monolithic":
            wrapped: nn.Module = MonolithicLoraLinear(
                base,
                rank=total_rank,
                alpha=alpha,
                dropout=dropout,
                groups=groups,
                controller=controller,
                orthogonal_init=orthogonal_init,
            )
        else:
            wrapped = StructuredLoraLinear(
                base,
                groups=groups,
                group_rank=group_rank,
                total_rank=total_rank,
                alpha=alpha,
                dropout=dropout,
                controller=controller,
                orthogonal_init=orthogonal_init,
            )
        setattr(parent, child_name, wrapped)
        replaced_names.append(name)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return controller, InjectionReport(method, targets, tuple(replaced_names), trainable, total)


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (name.endswith("lora_A") or name.endswith("lora_B"))
    }


def load_adapter_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    missing = sorted(set(state) - set(named))
    if missing:
        raise KeyError(f"adapter contains unknown parameters: {missing[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))

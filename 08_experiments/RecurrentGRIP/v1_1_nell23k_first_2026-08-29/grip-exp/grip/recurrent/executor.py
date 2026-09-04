from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
from torch import nn

from .config import resolve_layer_index


def _hidden_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported decoder block output: {type(output)!r}")


class FixedDepthRecurrentBlock(nn.Module):
    """Repeat one decoder block while sharing exactly the same parameters."""

    def __init__(self, block: nn.Module, depth: int = 1) -> None:
        super().__init__()
        self.block = block
        self.depth = 1
        self.record_trace = False
        self._last_trace: list[torch.Tensor] = []
        self.set_depth(depth)

    @property
    def attention_type(self) -> str:
        """Expose the decoder-layer attention type required by Transformers."""
        return self.block.attention_type

    def set_depth(self, depth: int) -> None:
        if depth < 1:
            raise ValueError("recurrent depth must be at least 1")
        self.depth = int(depth)

    def clear_trace(self) -> None:
        self._last_trace = []

    def get_trace(self) -> list[torch.Tensor]:
        return list(self._last_trace)

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        current = hidden_states
        output: Any = hidden_states
        trace: list[torch.Tensor] = []
        for _ in range(self.depth):
            output = self.block(current, *args, **kwargs)
            current = _hidden_from_output(output)
            if self.record_trace:
                trace.append(current[:, -1, :].detach().float().cpu())
        if self.record_trace:
            self._last_trace = trace
        return output


def _decoder_layer_candidates(model: nn.Module) -> list[tuple[str, nn.ModuleList]]:
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and (name == "layers" or name.endswith(".layers")):
            candidates.append((name, module))
    return candidates


def find_decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    candidates = _decoder_layer_candidates(model)
    if not candidates:
        raise ValueError("Could not locate a decoder ModuleList ending in '.layers'")
    candidates.sort(key=lambda item: (len(item[1]), item[0].count(".")), reverse=True)
    return candidates[0]


def wrap_decoder_layer(model: nn.Module, layer_index: int, depth: int) -> int:
    _, layers = find_decoder_layers(model)
    resolved = resolve_layer_index(len(layers), layer_index)
    existing = layers[resolved]
    if isinstance(existing, FixedDepthRecurrentBlock):
        existing.set_depth(depth)
    else:
        layers[resolved] = FixedDepthRecurrentBlock(existing, depth=depth)
    return resolved


def get_recurrent_block(model: nn.Module) -> FixedDepthRecurrentBlock:
    blocks = [module for module in model.modules() if isinstance(module, FixedDepthRecurrentBlock)]
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly one recurrent block, found {len(blocks)}")
    return blocks[0]


def set_recurrent_depth(model: nn.Module, depth: int) -> None:
    get_recurrent_block(model).set_depth(depth)


@contextmanager
def trace_recurrence(model: nn.Module) -> Iterator[FixedDepthRecurrentBlock]:
    block = get_recurrent_block(model)
    previous = block.record_trace
    block.clear_trace()
    block.record_trace = True
    try:
        yield block
    finally:
        block.record_trace = previous


def capture_recurrent_trace(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> list[list[float]]:
    """Run one cache-free prompt forward and return batch-0 pooled step states."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), trace_recurrence(model) as block:
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        return [step[0].tolist() for step in block.get_trace()]
    finally:
        model.train(was_training)

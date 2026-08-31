"""Pure-Python routing semantics for StructuredLoRA controls.

The functions in this module intentionally do not import torch so that the
causal structure can be audited on macOS before a CUDA environment is ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

METHODS = (
    "monolithic",
    "static_split",
    "flat_oracle",
    "ordered_prefix",
    "ordered_prefix_local_credit",
    "permuted_depth_prefix",
    "random_group_order",
    "non_nested_random_masks",
)


@dataclass(frozen=True)
class RouteMasks:
    """Forward activation and gradient-credit masks for one batch."""

    forward: tuple[tuple[float, ...], ...]
    credit: tuple[tuple[float, ...], ...]


def _one_hot(index: int, groups: int) -> tuple[float, ...]:
    return tuple(1.0 if group == index else 0.0 for group in range(groups))


def _prefix(depth: int, groups: int, order: Sequence[int] | None = None) -> tuple[float, ...]:
    order = tuple(range(groups)) if order is None else tuple(order)
    if sorted(order) != list(range(groups)):
        raise ValueError(f"group order must be a permutation of 0..{groups - 1}: {order}")
    active = set(order[:depth])
    return tuple(1.0 if group in active else 0.0 for group in range(groups))


def _validate_depths(depths: Iterable[int], groups: int) -> tuple[int, ...]:
    values = tuple(int(depth) for depth in depths)
    if not values:
        raise ValueError("depth batch must not be empty")
    invalid = [depth for depth in values if depth < 1 or depth > groups]
    if invalid:
        raise ValueError(f"depths must be within 1..{groups}: {invalid}")
    return values


def build_route_masks(
    method: str,
    depths: Iterable[int],
    *,
    groups: int = 4,
    depth_permutation: Mapping[int, int] | None = None,
    group_order: Sequence[int] | None = None,
    non_nested_masks: Mapping[int, Sequence[int]] | None = None,
    knockout_group: int | None = None,
) -> RouteMasks:
    """Build forward and gradient masks.

    ``ordered_prefix_local_credit`` uses all prefix groups in the forward pass,
    but only the newest group receives gradient for an example at depth d. The
    previous groups therefore act as frozen lower-depth functions on deeper
    examples, implementing depth-local credit assignment.
    """

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if groups < 1:
        raise ValueError("groups must be positive")
    values = _validate_depths(depths, groups)
    if knockout_group is not None and not 0 <= knockout_group < groups:
        raise ValueError(f"knockout_group must be in 0..{groups - 1}")

    permutation = dict(depth_permutation or {depth: depth for depth in range(1, groups + 1)})
    expected_depths = set(range(1, groups + 1))
    if set(permutation) != expected_depths or set(permutation.values()) != expected_depths:
        raise ValueError("depth_permutation must be a bijection over 1..groups")

    default_non_nested = {
        1: (2,),
        2: (0, 3),
        3: (0, 1, 3),
        4: (0, 1, 2, 3),
    }
    non_nested = dict(non_nested_masks or default_non_nested)

    forward_rows: list[tuple[float, ...]] = []
    credit_rows: list[tuple[float, ...]] = []
    for depth in values:
        if method == "monolithic":
            forward = tuple(1.0 for _ in range(groups))
            credit = forward
        elif method == "static_split":
            forward = tuple(1.0 for _ in range(groups))
            credit = forward
        elif method == "flat_oracle":
            forward = _one_hot(depth - 1, groups)
            credit = forward
        elif method == "ordered_prefix":
            forward = _prefix(depth, groups)
            credit = forward
        elif method == "ordered_prefix_local_credit":
            forward = _prefix(depth, groups)
            credit = _one_hot(depth - 1, groups)
        elif method == "permuted_depth_prefix":
            mapped_depth = permutation[depth]
            forward = _prefix(mapped_depth, groups)
            credit = forward
        elif method == "random_group_order":
            if group_order is None:
                raise ValueError("random_group_order requires an explicit fixed group_order")
            forward = _prefix(depth, groups, group_order)
            credit = forward
        elif method == "non_nested_random_masks":
            active = tuple(int(group) for group in non_nested[depth])
            if len(active) != depth or len(set(active)) != depth:
                raise ValueError(f"depth {depth} non-nested mask must contain {depth} unique groups")
            if any(group < 0 or group >= groups for group in active):
                raise ValueError(f"invalid group in non-nested mask for depth {depth}: {active}")
            active_set = set(active)
            forward = tuple(1.0 if group in active_set else 0.0 for group in range(groups))
            credit = forward
        else:  # pragma: no cover - guarded above
            raise AssertionError(method)

        if knockout_group is not None:
            forward = tuple(0.0 if group == knockout_group else value for group, value in enumerate(forward))
            credit = tuple(0.0 if group == knockout_group else value for group, value in enumerate(credit))
        if any(c > f for c, f in zip(credit, forward)):
            raise AssertionError("credit mask cannot activate a group absent from the forward pass")
        forward_rows.append(forward)
        credit_rows.append(credit)

    return RouteMasks(tuple(forward_rows), tuple(credit_rows))


def masks_are_nested(masks_by_depth: Mapping[int, Sequence[float]]) -> bool:
    """Return whether active sets form a cumulative chain from shallow to deep."""

    previous: set[int] = set()
    for depth in sorted(masks_by_depth):
        current = {i for i, value in enumerate(masks_by_depth[depth]) if value}
        if not previous.issubset(current):
            return False
        previous = current
    return True

"""Configuration loading and invariant checks."""

from __future__ import annotations

import json
from pathlib import Path

from .routing import METHODS, build_route_masks, masks_are_nested


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def _int_key_mapping(mapping: dict) -> dict[int, object]:
    return {int(key): value for key, value in mapping.items()}


def routing_kwargs(config: dict) -> dict:
    controls = config["routing_controls"]
    return {
        "depth_permutation": {int(key): int(value) for key, value in controls["depth_permutation"].items()},
        "group_order": tuple(int(value) for value in controls["random_group_order"]),
        "non_nested_masks": {
            int(key): tuple(int(value) for value in values)
            for key, values in controls["non_nested_masks"].items()
        },
    }


def validate_config(config: dict) -> None:
    if config.get("format_version") != 1:
        raise ValueError("format_version must be 1")
    lora = config["lora"]
    if lora["groups"] * lora["group_rank"] != lora["total_rank"]:
        raise ValueError("groups * group_rank must equal total_rank")
    if lora["groups"] != 4:
        raise ValueError("v0.2 exact-hop experiment requires four depth groups")
    methods = tuple(config["methods"])
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if len(methods) != len(set(methods)):
        raise ValueError("methods must be unique")
    required = {
        "monolithic",
        "static_split",
        "flat_oracle",
        "ordered_prefix",
        "ordered_prefix_local_credit",
        "permuted_depth_prefix",
        "random_group_order",
        "non_nested_random_masks",
    }
    missing = required - set(methods)
    if missing:
        raise ValueError(f"missing required controls: {sorted(missing)}")
    training = config["training"]
    if not training["seeds"] or not training["smoke_seeds"]:
        raise ValueError("seed lists must not be empty")
    if not set(training["smoke_seeds"]).issubset(training["seeds"]):
        raise ValueError("smoke_seeds must be a subset of seeds")
    if training["max_optimizer_steps"] <= 0:
        raise ValueError("max_optimizer_steps must be positive")

    kwargs = routing_kwargs(config)
    masks = {
        method: {
            depth: build_route_masks(method, [depth], groups=4, **kwargs).forward[0]
            for depth in range(1, 5)
        }
        for method in methods
    }
    if not masks_are_nested(masks["ordered_prefix"]):
        raise ValueError("ordered_prefix must be nested")
    if not masks_are_nested(masks["random_group_order"]):
        raise ValueError("fixed random_group_order remains nested and is a symmetry control")
    if masks_are_nested(masks["non_nested_random_masks"]):
        raise ValueError("non_nested_random_masks must actually break nesting")

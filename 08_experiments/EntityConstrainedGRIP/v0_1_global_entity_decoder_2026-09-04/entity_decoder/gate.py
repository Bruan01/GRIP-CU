"""Pre-registered validation-only Phase-A D0/D1 mechanism gate."""
from __future__ import annotations


def _metric(summary: dict, decoder: str, split: str, path: tuple[str, ...]) -> float:
    try:
        value = summary["results"][decoder][split]["metrics"]
        for key in path:
            value = value[key]
        return float(value)
    except (KeyError, TypeError, ValueError) as exc:
        joined = ".".join(path)
        raise ValueError(f"missing metric {decoder}/{split}/{joined}") from exc


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric list")
    return sum(values) / len(values)


def evaluate_gate(config: dict, summaries: dict[str, dict]) -> dict:
    """Evaluate D1 over imported D0 on validation across direct checkpoints."""
    gate = config["gate"]
    split = gate.get("split", "validation")
    decoder = gate.get("decoder", "D1")
    if split != "validation":
        raise ValueError("Phase-A decoder gate must use validation only")
    if decoder != "D1":
        raise ValueError("initial Phase-A gate must evaluate D1")

    primary_names = list(gate.get("primary_checkpoints", ()))
    reference_names = list(gate.get("reference_checkpoints", ()))
    minimum = int(gate.get("minimum_primary_checkpoints", 2))
    if len(primary_names) < minimum or minimum < 2:
        raise ValueError("gate requires at least two registered primary checkpoints")
    missing = [name for name in primary_names + reference_names if name not in summaries]
    if missing:
        raise ValueError(f"missing gate checkpoint summaries: {sorted(set(missing))}")

    per_checkpoint = {}
    for name in primary_names:
        summary = summaries[name]
        d0_canonical = _metric(summary, "D0", split, ("canonical_entity_exact_match",))
        d1_canonical = _metric(summary, decoder, split, ("canonical_entity_exact_match",))
        d0_raw = _metric(summary, "D0", split, ("raw_exact_match",))
        d1_raw = _metric(summary, decoder, split, ("raw_exact_match",))
        d0_valid = _metric(summary, "D0", split, ("valid_entity_rate",))
        d1_valid = _metric(summary, decoder, split, ("valid_entity_rate",))
        d0_novel = _metric(summary, "D0", split, ("by_composition", "novel-composition", "canonical_entity_exact_match"))
        d1_novel = _metric(summary, decoder, split, ("by_composition", "novel-composition", "canonical_entity_exact_match"))
        per_checkpoint[name] = {
            "canonical_gain_pp": 100.0 * (d1_canonical - d0_canonical),
            "raw_gain_pp": 100.0 * (d1_raw - d0_raw),
            "valid_rate_gain_pp": 100.0 * (d1_valid - d0_valid),
            "novel_composition_gain_pp": 100.0 * (d1_novel - d0_novel),
            "d0_canonical_em": d0_canonical,
            "d1_canonical_em": d1_canonical,
        }

    aggregate = {
        "mean_canonical_gain_pp": _mean([row["canonical_gain_pp"] for row in per_checkpoint.values()]),
        "mean_raw_gain_pp": _mean([row["raw_gain_pp"] for row in per_checkpoint.values()]),
        "mean_valid_rate_gain_pp": _mean([row["valid_rate_gain_pp"] for row in per_checkpoint.values()]),
        "mean_novel_composition_gain_pp": _mean([row["novel_composition_gain_pp"] for row in per_checkpoint.values()]),
    }
    same_direction = all(
        row["canonical_gain_pp"] >= 0.0 and row["raw_gain_pp"] >= 0.0
        for row in per_checkpoint.values()
    )
    checks = {
        "minimum_primary_checkpoints": len(per_checkpoint) >= minimum,
        "mean_canonical_gain": aggregate["mean_canonical_gain_pp"] >= float(gate["mean_canonical_gain_pp"]),
        "raw_and_canonical_same_direction": same_direction if gate.get("require_nonnegative_per_checkpoint_raw_and_canonical", True) else True,
        "novel_composition_floor": aggregate["mean_novel_composition_gain_pp"] >= -float(gate["max_mean_novel_composition_drop_pp"]),
    }
    passed = all(checks.values())

    reference_diagnostics = {}
    for name in reference_names:
        summary = summaries[name]
        reference_diagnostics[name] = {
            "canonical_gain_pp": 100.0 * (
                _metric(summary, decoder, split, ("canonical_entity_exact_match",))
                - _metric(summary, "D0", split, ("canonical_entity_exact_match",))
            ),
            "valid_rate_gain_pp": 100.0 * (
                _metric(summary, decoder, split, ("valid_entity_rate",))
                - _metric(summary, "D0", split, ("valid_entity_rate",))
            ),
        }

    return {
        "decision": "PRELIMINARY_GO_D2" if passed else "STOP_DECODER_PRIMARY",
        "split": split,
        "decoder": decoder,
        "primary_checkpoints": primary_names,
        "reference_checkpoints": reference_names,
        "checks": checks,
        "per_checkpoint": per_checkpoint,
        "aggregate": aggregate,
        "reference_diagnostics": reference_diagnostics,
        "next_action": "RUN_D2_VALIDATION_ONLY" if passed else "STOP_D2_AND_PHASE_B",
    }

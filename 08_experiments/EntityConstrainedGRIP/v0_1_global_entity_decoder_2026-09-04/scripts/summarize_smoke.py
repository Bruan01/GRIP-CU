#!/usr/bin/env python3
"""Combine checkpoint summaries and apply the validation-only D0/D1 gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entity_decoder.config import load_config
from entity_decoder.gate import evaluate_gate
from entity_decoder.io_utils import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/phase_a_decoder_smoke.json")
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    config = load_config(args.config.resolve())
    summaries: dict[str, dict] = {}
    summary_paths: dict[str, list[str]] = {}
    for path in sorted(args.run_dir.resolve().rglob("run_summary.json")):
        current = json.loads(path.read_text(encoding="utf-8"))
        name = current["checkpoint"]["name"]
        summary_paths.setdefault(name, []).append(str(path))
        if name not in summaries:
            summaries[name] = current
            continue
        merged = summaries[name]
        if merged["prompt_protocol"] != current["prompt_protocol"]:
            raise ValueError(f"cannot merge prompt protocols for {name}")
        for decoder, split_results in current["results"].items():
            if decoder in merged["results"]:
                raise ValueError(f"duplicate decoder result for {name}/{decoder}")
            merged["results"][decoder] = split_results
        merged.setdefault("mechanism_probes", {}).update(current.get("mechanism_probes", {}))

    gate = config["gate"]
    required = set(gate["primary_checkpoints"] + gate.get("reference_checkpoints", []))
    missing = sorted(required - set(summaries))
    if missing:
        raise FileNotFoundError(f"missing gate checkpoint summaries: {missing}")
    decision = evaluate_gate(config, summaries)
    output = {
        "experiment_id": config["experiment_id"],
        "protocol": "validation_only_d0_artifact_reuse_vs_d1_global_trie",
        "checkpoint_summaries": sorted(summaries),
        "summary_paths": summary_paths,
        "gate": decision,
    }
    write_json(args.run_dir.resolve() / "suite_summary.json", output)

    report = [
        "# EntityConstrained-GRIP Phase-A Validation Mechanism Audit",
        "",
        f"- Decision: **{decision['decision']}**",
        f"- Split: **{decision['split']} only**",
        f"- Decoder comparison: **D0 frozen artifact → {decision['decoder']} global train-KG trie**",
        f"- Next action: `{decision['next_action']}`",
        "- Test split: **not opened by the Phase-A runner**",
        "",
        "## Gate checks",
        "",
    ]
    report.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in decision["checks"].items())
    report.extend(["", "## Aggregate gains across direct checkpoints", ""])
    report.extend(f"- {name}: {value:.3f} pp" for name, value in decision["aggregate"].items())
    report.extend([
        "",
        "## Direct checkpoint gains",
        "",
        "| checkpoint | canonical | raw | valid-rate | novel-composition |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, values in decision["per_checkpoint"].items():
        report.append(
            f"| {name} | {values['canonical_gain_pp']:.3f} | {values['raw_gain_pp']:.3f} | "
            f"{values['valid_rate_gain_pp']:.3f} | {values['novel_composition_gain_pp']:.3f} |"
        )
    report.extend(["", "## D0→D1 error transitions", ""])
    for name in decision["primary_checkpoints"]:
        transitions = summaries[name].get("mechanism_probes", {}).get("d0_to_d1_validation_transitions")
        if not transitions:
            report.append(f"- {name}: missing")
            continue
        counts = transitions["counts"]
        report.append(
            f"- {name}: invalid→correct={counts['d0_invalid_to_d1_correct']}, "
            f"valid-wrong→correct={counts['d0_valid_wrong_to_d1_correct']}, "
            f"correct→wrong={counts['d0_correct_to_d1_wrong']}, "
            f"wrong→wrong={counts['d0_wrong_unchanged']}"
        )
    (args.run_dir.resolve() / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(decision["decision"])


if __name__ == "__main__":
    main()

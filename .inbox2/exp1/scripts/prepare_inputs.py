#!/usr/bin/env python3
"""Copy bounded NELL23K GRIP task samples into the isolated exp1 inputs tree."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / "08_experiments/Hard_Negative_Contrastive_GRIP/grip_nell23k_tasks.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-stage", type=int, default=16)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    selected = {}
    for name in ("context_samples", "qa_samples"):
        rows = payload.get(name)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing non-empty {name}")
        n = min(args.samples_per_stage, len(rows))
        # A deterministic shuffled subset avoids selecting only the file prefix.
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        selected[name] = [rows[i] for i in sorted(indices[:n])]
    input_dir = ROOT / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied_source = input_dir / "source_tasks.json"
    shutil.copy2(source, copied_source)
    smoke_file = input_dir / f"nell23k_smoke_{args.samples_per_stage}_each.json"
    temporary = smoke_file.with_suffix(smoke_file.suffix + ".tmp")
    output = {
        "format_version": 1,
        "seed": args.seed,
        "source_file": source.name,
        "source_sha256": sha256(source),
        "selected_counts": {key: len(value) for key, value in selected.items()},
        "context_samples": selected["context_samples"],
        "qa_samples": selected["qa_samples"],
    }
    temporary.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    temporary.replace(smoke_file)
    manifest = {
        "source_readonly_path": str(source),
        "source_sha256": sha256(source),
        "local_source_copy": str(copied_source.relative_to(ROOT)),
        "local_source_copy_sha256": sha256(copied_source),
        "smoke_input": str(smoke_file.relative_to(ROOT)),
        "seed": args.seed,
        "selected_counts": output["selected_counts"],
        "notice": "Smoke data copied and sampled within exp1; source was read-only.",
    }
    (ROOT / "manifest").mkdir(exist_ok=True)
    (ROOT / "manifest" / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

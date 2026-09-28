#!/usr/bin/env python3
"""Download a pinned Qwen2.5-0.5B-Instruct snapshot inside exp1 only."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    destination = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(destination),
        cache_dir=str(ROOT / ".hf_cache"),
        resume_download=True,
    )
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (destination / name).is_file()]
    if not list(destination.glob("*.safetensors")) and not (destination / "model.safetensors.index.json").is_file():
        missing.append("model weights (*.safetensors or index)")
    if missing:
        raise RuntimeError(f"Model snapshot incomplete at {result}; missing {missing}")
    print(f"Model snapshot ready in exp1: {destination}")
    print(f"Repository: {args.repo_id}@{args.revision}")


if __name__ == "__main__":
    main()

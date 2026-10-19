"""Download a GRIP base model from ModelScope with verified, resumable shards.

The ModelScope SDK is used for small metadata files.  Multi-GB safetensors
shards are fetched serially with curl range-resume and SHA-256 validation,
which is more reliable than the SDK parallel downloader on Windows/proxied
connections.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import get_file_download_url
from modelscope import snapshot_download

from constants import HF_DECODER_ONLY_LLMS, MODELSCOPE_DECODER_ONLY_LLMS
from models.utils import _is_complete_model_dir


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_shard(path: Path, expected_size: int, expected_sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    return sha256(path).lower() == expected_sha256.lower()


def download_shard(model_id: str, model_dir: Path, metadata: dict, retries: int, log_dir: Path | None = None) -> None:
    name, size, expected_sha256, revision = (
        metadata["Name"], metadata["Size"], metadata["Sha256"], metadata["Revision"]
    )
    final_path = model_dir / name
    if validate_shard(final_path, size, expected_sha256):
        print(f"Verified existing shard: {name}")
        return
    if final_path.exists():
        print(f"Removing invalid completed shard: {name}")
        final_path.unlink()

    temp_dir = model_dir / ".curl-partials"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{name}.part"
    url = get_file_download_url(model_id, name, revision, endpoint="https://www.modelscope.cn")

    for attempt in range(1, retries + 1):
        if temp_path.exists() and temp_path.stat().st_size > size:
            temp_path.unlink()
        resume_size = temp_path.stat().st_size if temp_path.exists() else 0
        print(f"Shard {name}: attempt {attempt}/{retries}; resuming at {resume_size:,}/{size:,} bytes")
        command = [
            shutil.which("curl.exe") or shutil.which("curl") or "curl",
            "--fail", "--location", "--continue-at", "-",
            "--retry", "8", "--retry-all-errors", "--connect-timeout", "45",
            "--output", str(temp_path), url,
        ]
        if log_dir is None:
            result = subprocess.run(command, check=False)
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / f"{name}.curl.log").open("ab") as log_file:
                result = subprocess.run(command, check=False, stdout=log_file, stderr=log_file)
        if result.returncode == 0 and validate_shard(temp_path, size, expected_sha256):
            temp_path.replace(final_path)
            print(f"Verified shard: {name}")
            return

        actual_size = temp_path.stat().st_size if temp_path.exists() else 0
        print(f"Shard validation failed (curl={result.returncode}, size={actual_size:,}); retrying from scratch.")
        temp_path.unlink(missing_ok=True)

    raise RuntimeError(f"Unable to obtain verified shard after {retries} attempts: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", choices=sorted(HF_DECODER_ONLY_LLMS), default="qwen-7b")
    parser.add_argument("--model_cache_dir", default="model_cache")
    parser.add_argument("--model_download_workers", type=int, default=1,
                        help="Workers for small metadata files; weight shards are always serial.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--log_dir", default="artifacts/download_logs")
    parser.add_argument("--shards", nargs="+", default=None,
                        help="Optional safetensors shard names; omit to download all missing shards.")
    args = parser.parse_args()

    model_id = MODELSCOPE_DECODER_ONLY_LLMS[args.model_name]
    model_dir = Path(args.model_cache_dir) / model_id.replace("/", "--")
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    index_path = model_dir / "model.safetensors.index.json"
    # Never re-run the ModelScope SDK against an existing partial weight cache:
    # it may attempt a concurrent cache move on Windows. The metadata files are
    # immutable for the pinned revision and only need downloading once.
    if not index_path.is_file():
        snapshot_download(
            model_id=model_id,
            local_dir=str(model_dir),
            max_workers=args.model_download_workers,
            allow_patterns=["*.json", "*.txt", "*.md", "LICENSE", "tokenizer*", "vocab.*", "merges.*"],
        )
    if not index_path.is_file():
        raise RuntimeError(f"Missing model weight index after metadata download: {index_path}")

    api_files = HubApi().get_model_files(model_id, revision="master", recursive=True)
    shard_metadata = {entry["Name"]: entry for entry in api_files if entry["Name"].endswith(".safetensors")}
    weight_names = sorted(set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()))
    selected = args.shards or weight_names
    unknown = sorted(set(selected) - set(weight_names))
    if unknown:
        raise ValueError(f"Unknown shard name(s): {unknown}. Available: {weight_names}")

    for name in selected:
        download_shard(model_id, model_dir, shard_metadata[name], args.retries, Path(args.log_dir))

    if args.shards is None:
        if not _is_complete_model_dir(model_dir):
            missing = [name for name in weight_names if not (model_dir / name).is_file()]
            raise RuntimeError(f"Download is incomplete at {model_dir}; missing {missing}")
        print(f"Model ready: {model_dir.resolve()}")
    else:
        print(f"Verified selected shard(s): {', '.join(selected)}")


if __name__ == "__main__":
    main()

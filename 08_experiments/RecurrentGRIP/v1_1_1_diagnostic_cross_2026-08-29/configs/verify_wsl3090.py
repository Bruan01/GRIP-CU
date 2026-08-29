from __future__ import annotations

import json
import platform
import sys

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA check failed: torch.cuda.is_available() is false")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device_index": index,
        "device_name": properties.name,
        "device_memory_gib": round(properties.total_memory / 2**30, 2),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if "3090" not in properties.name:
        raise SystemExit(f"expected an RTX 3090, detected: {properties.name}")
    if properties.total_memory < 20 * 2**30:
        raise SystemExit("detected GPU memory is below the expected 3090-class 24 GiB")
    if capability < (8, 0):
        raise SystemExit(f"unexpected compute capability: {capability}")


if __name__ == "__main__":
    main()

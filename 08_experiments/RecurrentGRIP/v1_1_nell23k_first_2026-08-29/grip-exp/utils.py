import json
import os
import socket
import sys
import os.path as osp
import random
import shutil
import time
import zipfile
from typing import Any, Union, Optional

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from transformers import set_seed
import re


def _json_safe(value: Any) -> Any:
    """Convert common experiment/config objects into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "dtype") and hasattr(value, "device"):
        return str(value)
    if hasattr(value, "__dict__"):
        return {str(key): _json_safe(item) for key, item in vars(value).items()
                if not str(key).startswith("_")}
    return str(value)


def log_event(event: str, level: str = "INFO", **fields: Any) -> None:
    """Print a timestamped, structured event suitable for experiment provenance.

    Bash launchers capture stdout with ``tee``; keeping this logger stdout-only
    means Python and subprocess logs end up in the same reproducible run log.
    Every line contains process/rank/runtime context so multi-GPU output can be
    disentangled after the run.
    """
    context = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "event": event,
    }
    context.update({key: _json_safe(value) for key, value in fields.items()})
    print(f"[{context.pop('timestamp')}] [{level.upper()}] "
          f"event={event} {json.dumps(context, ensure_ascii=False, sort_keys=True)}",
          flush=True)


def runtime_summary() -> dict[str, Any]:
    """Return Python, PyTorch, CUDA and GPU metadata for run provenance."""
    result: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    if torch.cuda.is_available():
        result["gpu_count"] = torch.cuda.device_count()
        result["gpus"] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["gpus"].append({
                "index": index,
                "name": props.name,
                "total_memory_gib": round(props.total_memory / 2**30, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    else:
        result["gpu_count"] = 0
    return result


def log_runtime(event: str = "runtime_environment") -> None:
    log_event(event, **runtime_summary())


def model_summary(model: Any) -> dict[str, Any]:
    """Summarize model identity, size, dtype, placement and trainable weights."""
    total = trainable = 0
    dtypes: set[str] = set()
    devices: set[str] = set()
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
        dtypes.add(str(parameter.dtype))
        devices.add(str(parameter.device))
    summary = {
        "class": model.__class__.__name__,
        "model_config_type": getattr(getattr(model, "config", None), "model_type", None),
        "parameters": total,
        "parameters_million": round(total / 1e6, 3),
        "trainable_parameters": trainable,
        "trainable_parameters_million": round(trainable / 1e6, 3),
        "trainable_percent": round(100 * trainable / total, 6) if total else 0.0,
        "dtypes": sorted(dtypes),
        "devices": sorted(devices),
    }
    try:
        summary["memory_footprint_gib"] = round(model.get_memory_footprint() / 2**30, 3)
    except Exception:
        pass
    return summary


def log_cuda_memory(event: str = "cuda_memory") -> None:
    if not torch.cuda.is_available():
        log_event(event, cuda_available=False)
        return
    devices = []
    for index in range(torch.cuda.device_count()):
        devices.append({
            "index": index,
            "allocated_gib": round(torch.cuda.memory_allocated(index) / 2**30, 3),
            "reserved_gib": round(torch.cuda.memory_reserved(index) / 2**30, 3),
            "max_allocated_gib": round(torch.cuda.max_memory_allocated(index) / 2**30, 3),
        })
    log_event(event, cuda_available=True, devices=devices)


def extract_tag_content(text: str, tag: str):
    pattern = fr"<{tag}>(.*?)</{tag}>"
    return re.findall(pattern, text)


def save_json(file_path: str, data: Any, mode="w"):
    with open(file_path, mode, encoding='utf-8') as f:
        json.dump(data, f)


def save_list_json(file_path: str, data: Any):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def save_entry_to_list_json(file_path: str, data: Any):
    with open(file_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def load_json(file_path: str) -> Union[dict, list[dict]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_list_json(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def make_dir(dir_name: str):
    os.makedirs(dir_name, exist_ok=True)


def is_main_process():
    try:
        # Hugging Face Accelerate
        accelerator = Accelerator()
        return accelerator.is_main_process
    except ImportError:
        pass  # Accelerate not installed

    try:
        # PyTorch Distributed
        return not dist.is_initialized() or dist.get_rank() == 0
    except ImportError:
        pass  # PyTorch not installed

    # Default to True for single-process execution
    return True


def download_hf_file(repo_id,
                     filename,
                     local_dir,
                     subfolder=None,
                     repo_type="dataset",
                     cache_dir=None,
                     ) -> str:
    hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename, repo_type=repo_type,
                    local_dir=local_dir, local_dir_use_symlinks=False, cache_dir=cache_dir, force_download=True)
    if subfolder is not None:
        shutil.move(osp.join(local_dir, subfolder, filename), osp.join(local_dir, filename))
        shutil.rmtree(osp.join(local_dir, subfolder))
    return osp.join(local_dir, filename)


def extract_zip(path: str, folder: str):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def gen_random_seed() -> int:
    return int(time.time() * 1000) % (2 ** 32 - 1)


def set_random_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = gen_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    log_event("random_seed_set", seed=seed)
    return seed


class Timer:
    def __init__(self):
        self.total_time = 0.0
        self._start_time = None

    def start(self):
        if self._start_time is None:
            self._start_time = time.time()
        else:
            raise RuntimeError("Timer is already running. Call end() before calling start() again.")

    def end(self):
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self.total_time += elapsed
            self._start_time = None
        else:
            raise RuntimeError("Timer is not running. Call start() before calling end().")

    def return_time(self):
        current_total = self.total_time
        if self._start_time is not None:
            # Include time since last start
            current_total += time.time() - self._start_time
        return current_total

    def reset(self):
        self.total_time = 0.0
        self._start_time = None
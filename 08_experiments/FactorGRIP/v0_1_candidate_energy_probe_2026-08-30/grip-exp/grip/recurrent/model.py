from __future__ import annotations

from pathlib import Path
from typing import Optional

from peft import LoraConfig, TaskType, get_peft_model

from constants import HF_DECODER_ONLY_LLMS, MODELSCOPE_DECODER_ONLY_LLMS, TORCH_DTYPE
from models.utils import get_hf_llm_tokenizer

from .executor import find_decoder_layers, wrap_decoder_layer


def _model_id(logical_name: str, model_source: str) -> str:
    # ``resolve_model_path`` accepts an explicit local Transformers directory.
    # Keep that path intact instead of treating it as a logical alias; this is
    # needed for standard Hugging Face snapshot caches outside ``model_cache``.
    explicit_path = Path(logical_name).expanduser()
    if explicit_path.is_dir():
        return str(explicit_path)
    table = MODELSCOPE_DECODER_ONLY_LLMS if model_source in {"auto", "modelscope"} else HF_DECODER_ONLY_LLMS
    if logical_name not in table:
        raise KeyError(f"Unknown model alias: {logical_name}")
    return table[logical_name]


def validate_adapter_scope(model, executor_layer_index: int) -> list[str]:
    """Verify that every trainable parameter is a LoRA tensor inside the executor layer."""
    marker = f".layers.{executor_layer_index}."
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable adapter parameters were found")
    outside = [name for name in trainable if "lora_" not in name or marker not in f".{name}"]
    if outside:
        raise ValueError(
            "Trainable parameters escaped the recurrent executor layer: " + ", ".join(outside[:8])
        )
    return trainable


def build_recurrent_peft_model(
    model_name: str,
    model_source: str = "auto",
    model_cache_dir: str = "model_cache",
    local_files_only: bool = False,
    model_download_workers: int = 1,
    padding_side: str = "left",
    truncation_side: str = "left",
    tokenize_max_length: int = 4096,
    quantization: bool = False,
    dtype: str = "bfloat16",
    lora_r: int = 4,
    lora_alpha: int = 32,
    target_modules: Optional[list[str]] = None,
    dropout: float = 0.0,
    executor_layer_index: int = -1,
    recurrent_depth_train: int = 2,
    adapter_name: str = "default",
    device_map=None,
    **_: object,
):
    """Load a base LM, wrap one decoder layer recurrently, and attach layer-local LoRA."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj"]
    if dtype not in TORCH_DTYPE:
        raise ValueError(f"Unsupported dtype: {dtype}")
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=_model_id(model_name, model_source),
        model_source=model_source,
        model_cache_dir=model_cache_dir,
        local_files_only=local_files_only,
        model_download_workers=model_download_workers,
        device_map=device_map,
        padding_side=padding_side,
        truncation_side=truncation_side,
        tokenize_max_length=tokenize_max_length,
        quantization=quantization,
        dtype=TORCH_DTYPE[dtype],
        peft=False,
    )
    _, layers = find_decoder_layers(base_model)
    resolved_index = wrap_decoder_layer(base_model, executor_layer_index, recurrent_depth_train)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        layers_to_transform=[resolved_index],
        layers_pattern="layers",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config, adapter_name=adapter_name)
    model.config.use_cache = False
    validate_adapter_scope(model, resolved_index)
    return model, tokenizer, resolved_index

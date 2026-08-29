from pathlib import Path
from typing import Optional, Union
import json
import time

import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PrefixTuningConfig, PeftMixedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel

from utils import log_event, model_summary, log_cuda_memory


def _is_complete_model_dir(model_dir: Path) -> bool:
    """Return whether a local Transformers model directory has all weight shards."""
    if not (model_dir / "config.json").is_file():
        return False
    index_files = (
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    )
    for index_file in index_files:
        if index_file.is_file():
            import json

            try:
                weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
            except (OSError, ValueError, KeyError):
                return False
            return all((model_dir / name).is_file() and (model_dir / name).stat().st_size > 0
                       for name in set(weight_map.values()))
    return any((model_dir / name).is_file() and (model_dir / name).stat().st_size > 0
               for name in ("model.safetensors", "pytorch_model.bin"))


def resolve_model_path(
        model_id: str,
        model_source: str = "auto",
        model_cache_dir: Optional[str] = "model_cache",
        local_files_only: bool = False,
        model_download_workers: int = 1,
) -> str:
    """Resolve a Hugging Face-compatible model path with ModelScope-first mirroring.

    ``auto`` uses a complete local directory first, then downloads from
    ModelScope (the preferred mirror), and finally falls back to Hugging Face.
    Incomplete local downloads are resumed rather than mistaken for usable
    checkpoints.  One worker is the safest setting for multi-GB shards on Windows.
    """
    started = time.time()
    log_event("model_resolve_start", model_id=model_id, model_source=model_source,
              model_cache_dir=str(Path(model_cache_dir or "model_cache").expanduser().resolve()),
              local_files_only=local_files_only, model_download_workers=model_download_workers)
    if model_source not in {"auto", "local", "hf", "modelscope"}:
        raise ValueError("model_source must be one of: auto, local, hf, modelscope")
    if model_download_workers < 1:
        raise ValueError("model_download_workers must be at least 1")

    path = Path(model_id).expanduser()
    if path.is_dir():
        if _is_complete_model_dir(path):
            resolved = str(path.resolve())
            log_event("model_resolve_complete", model_id=model_id, resolved_path=resolved,
                      source="explicit_local_path", elapsed_seconds=round(time.time() - started, 3))
            return resolved
        log_event("model_resolve_incomplete", level="ERROR", model_id=model_id, path=str(path.resolve()))
        raise FileNotFoundError(f"Local model directory is incomplete: {path}")

    # The named model ID may use either its Hugging Face or ModelScope alias.
    # Look for both equivalent cache-directory spellings before consulting a hub.
    cache_dir = Path(model_cache_dir or "model_cache").expanduser()
    local_candidates = [cache_dir / model_id.replace("/", "--")]
    try:
        from constants import HF_DECODER_ONLY_LLMS, MODELSCOPE_DECODER_ONLY_LLMS

        for logical_name, hf_id in HF_DECODER_ONLY_LLMS.items():
            mirror_id = MODELSCOPE_DECODER_ONLY_LLMS.get(logical_name)
            if model_id in {hf_id, mirror_id}:
                local_candidates.extend(
                    cache_dir / candidate.replace("/", "--")
                    for candidate in (hf_id, mirror_id) if candidate is not None
                )
    except ImportError:  # pragma: no cover - resolver can be used independently.
        pass
    unique_candidates = list(dict.fromkeys(local_candidates))
    log_event("model_local_candidates", model_id=model_id,
              candidates=[str(candidate.resolve()) for candidate in unique_candidates])
    for local_dir in unique_candidates:
        if _is_complete_model_dir(local_dir):
            resolved = str(local_dir.resolve())
            index_files = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.bin"))
            total_bytes = sum(item.stat().st_size for item in index_files if item.is_file())
            log_event("model_resolve_complete", model_id=model_id, resolved_path=resolved,
                      source="local_cache", weight_files=len(index_files),
                      total_weight_gib=round(total_bytes / 2**30, 3),
                      elapsed_seconds=round(time.time() - started, 3))
            return resolved

    if model_source == "local" or local_files_only:
        searched = ", ".join(str(candidate) for candidate in unique_candidates)
        log_event("model_resolve_failed_local", level="ERROR", model_id=model_id, searched=searched,
                  elapsed_seconds=round(time.time() - started, 3))
        raise FileNotFoundError(
            f"Base model is not available locally: {model_id}. Searched: {searched}. "
            "Download it first with --model_source modelscope, or disable --local_files_only."
        )

    local_dir = local_candidates[0]
    errors = []
    if model_source in {"auto", "modelscope"}:
        try:
            from modelscope import snapshot_download

            cache_dir.mkdir(parents=True, exist_ok=True)
            log_event("model_download_start", model_id=model_id, source="modelscope",
                      destination=str(local_dir.resolve()))
            resolved = snapshot_download(
                model_id=model_id,
                local_dir=str(local_dir),
                max_workers=model_download_workers,
            )
            log_event("model_download_complete", model_id=model_id, source="modelscope",
                      resolved_path=str(Path(resolved).resolve()),
                      elapsed_seconds=round(time.time() - started, 3))
            return resolved
        except Exception as error:
            errors.append(f"ModelScope: {type(error).__name__}: {error}")
            if model_source == "modelscope":
                raise RuntimeError(
                    f"Failed to download {model_id} from ModelScope. " + " | ".join(errors)
                ) from error

    if model_source in {"auto", "hf"}:
        # Let transformers/huggingface_hub resolve this identifier on first load.
        log_event("model_resolve_remote", model_id=model_id, source="huggingface",
                  resolved_path=model_id, elapsed_seconds=round(time.time() - started, 3))
        return model_id

    raise RuntimeError(f"Could not resolve base model {model_id}. " + " | ".join(errors))


def get_hf_tokenizer(
        model_name: str,
        model_source: str = "auto",
        model_cache_dir: Optional[str] = "model_cache",
        local_files_only: bool = False,
        model_download_workers: int = 1,
        padding_side: str = "left",
        truncation_side: str = "left",
        tokenize_max_length: int = 4096,
        use_fast: bool = True,
        **kwargs,
) -> PreTrainedTokenizer:
    """Load only a tokenizer, avoiding duplicate base-model residency during task generation."""
    started = time.time()
    log_event("tokenizer_load_start", model_name=model_name, model_source=model_source,
              local_files_only=local_files_only, tokenize_max_length=tokenize_max_length)
    model_path = resolve_model_path(
        model_name,
        model_source=model_source,
        model_cache_dir=model_cache_dir,
        local_files_only=local_files_only,
        model_download_workers=model_download_workers,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast,
        padding=True,
        truncation=True,
        padding_side=padding_side,
        truncation_side=truncation_side,
        max_length=tokenize_max_length,
        local_files_only=local_files_only,
        **kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_event("tokenizer_load_complete", model_name=model_name, resolved_path=str(Path(model_path).resolve()) if Path(model_path).exists() else model_path,
              tokenizer_class=tokenizer.__class__.__name__, vocab_size=len(tokenizer),
              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
              padding_side=tokenizer.padding_side, truncation_side=tokenizer.truncation_side,
              elapsed_seconds=round(time.time() - started, 3))
    return tokenizer


def get_hf_llm_tokenizer(
        model_name: str,
        load_dir: Optional[str] = None,
        model_source: str = "auto",
        model_cache_dir: Optional[str] = "model_cache",
        local_files_only: bool = False,
        model_download_workers: int = 1,
        device_map: Optional[str] = "auto",
        padding_side: str = "left",
        truncation_side: str = "left",
        tokenize_max_length: int = 4096,
        use_fast: bool = True,
        flash_attention: bool = False,
        quantization: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        peft: bool = False,
        use_prefix: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        target_modules: Optional[list[str]] = None,
        dropout: float = 0.0,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128,
        **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    started = time.time()
    requested_model_name = model_name
    log_event("model_load_start", model_name=requested_model_name, model_source=model_source,
              load_dir=str(Path(load_dir).expanduser().resolve()) if load_dir else None,
              device_map=device_map, dtype=str(dtype), quantization=quantization,
              peft=peft, flash_attention=flash_attention,
              lora_r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules)
    if peft:
        device_map = None

    model_name = resolve_model_path(
        model_name,
        model_source=model_source,
        model_cache_dir=model_cache_dir,
        local_files_only=local_files_only,
        model_download_workers=model_download_workers,
    )

    if flash_attention:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        bnb_config = None
    # Load the adapter tokenizer before attaching the adapter.  PEFT may save
    # resized embedding/lm_head tensors alongside LoRA weights (as it does for
    # this project), so the base model must have the same vocabulary size before
    # ``PeftModel.from_pretrained`` calls ``load_state_dict``.
    tokenizer_source = load_dir if load_dir is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=use_fast,
        padding=True,
        truncation=True,
        padding_side=padding_side,
        truncation_side=truncation_side,
        max_length=tokenize_max_length,
        local_files_only=local_files_only,
        **kwargs,
    )
    pad_token = "[PAD]"
    if pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": pad_token})
    else:
        tokenizer.pad_token = pad_token

    log_event("model_weights_loading", model_name=requested_model_name, resolved_path=model_name,
              device_map=device_map, torch_dtype=str(dtype), attention_implementation=attn_implementation,
              quantization=bool(bnb_config))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )

    log_event("model_weights_loaded", model_name=requested_model_name, resolved_path=model_name,
              elapsed_seconds=round(time.time() - started, 3), **model_summary(model))
    log_cuda_memory("model_memory_after_base_load")

    model_vocab_size = model.get_input_embeddings().num_embeddings
    tokenizer_vocab_size = len(tokenizer)
    if model_vocab_size != tokenizer_vocab_size:
        log_event("model_resize_token_embeddings", model_name=requested_model_name,
                  model_vocab_size=model_vocab_size, tokenizer_vocab_size=tokenizer_vocab_size,
                  reason="align_base_model_with_tokenizer_before_adapter_load" if load_dir is not None
                  else "align_base_model_with_tokenizer")
        model.resize_token_embeddings(tokenizer_vocab_size)

    if load_dir is not None:
        log_event("adapter_load_start", adapter_dir=str(Path(load_dir).expanduser().resolve()),
                  tokenizer_vocab_size=tokenizer_vocab_size)
        model = PeftModel.from_pretrained(
            model,
            load_dir,
            torch_dtype=dtype,
            **kwargs,
        )
        model = model.merge_and_unload()
        log_event("adapter_load_complete", adapter_dir=str(Path(load_dir).expanduser().resolve()),
                  **model_summary(model))

    if peft:
        log_event("peft_attach_start", use_prefix=use_prefix, lora_r=lora_r,
                  lora_alpha=lora_alpha, target_modules=target_modules, dropout=dropout)
        model = load_peft_model(
            model,
            use_prefix=use_prefix,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            dropout=dropout,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
            **kwargs,
        )
        log_event("peft_attach_complete", **model_summary(model))
    log_event("tokenizer_ready", model_name=requested_model_name, tokenizer_class=tokenizer.__class__.__name__,
              vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
    log_cuda_memory("model_load_complete_memory")
    log_event("model_load_complete", model_name=requested_model_name, resolved_path=model_name,
              elapsed_seconds=round(time.time() - started, 3), **model_summary(model))
    return model, tokenizer


def load_peft_model(
        model: PreTrainedModel,
        use_prefix: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        target_modules: Optional[list[str]] = None,
        dropout: float = 0.0,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128,
        **kwargs
) -> PeftModel:
    if use_prefix:
        print("Load prefix tuning model......")
        model = get_prefix_model(
            model,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
            **kwargs,
        )
    else:
        print("Load LoRA model......")
        model = get_lora_model(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            dropout=dropout,
            **kwargs,
        )
    return model


def get_lora_model(
        model: Union[PreTrainedModel, PeftModel],
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        **kwargs) -> PeftModel:
    if target_modules is None:
        target_modules = ["k_proj", "v_proj", "q_proj"]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    # Remove previous lora modules.
    while isinstance(model, PeftModel):
        if hasattr(model, "unload"):
            model = model.unload()
        else:
            model = model.base_model
        
    model = get_peft_model(model, lora_config, adapter_name="mylora")
    model.print_trainable_parameters()

    return model


def get_prefix_model(
        model: Union[PreTrainedModel, PeftModel],
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128,
        **kwargs) -> PeftModel:
    r"""Create prefix model.
    """
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
        prefix_projection=True,

    )
    # Remove previous lora modules.
    while isinstance(model, PeftModel) or isinstance(model, PeftMixedModel):
        if hasattr(model, "unload"):
            model = model.unload()
        else:
            model = model.base_model
    model = get_peft_model(model, prefix_config, adapter_name="myprefix")
    model.print_trainable_parameters()

    return model

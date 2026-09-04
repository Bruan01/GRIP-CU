"""WSL/CUDA runtime loading and D0/D1 decoding."""
from __future__ import annotations

import time
from pathlib import Path

from .lora import inject_lora, load_adapter_state_dict
from .records import build_prompt
from .trie import EntityTokenTrie, TrieConstraint


def _dtype(name: str):
    import torch
    values = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"unsupported dtype: {name}")
    return values[name]


def load_runtime(config: dict, checkpoint: dict, model_override: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    model_name = model_override or model_config["name_or_path"]
    device = torch.device(model_config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered Phase-A run")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=bool(model_config.get("local_files_only", False)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer requires eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = _dtype(model_config.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=bool(model_config.get("local_files_only", False)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        torch_dtype=dtype,
        attn_implementation=model_config.get("attn_implementation", "eager"),
    ).to(device)
    injection = None
    if checkpoint["kind"] == "adapter":
        injection = inject_lora(model, **config["lora"])
        state = torch.load(Path(checkpoint["resolved_path"]), map_location="cpu", weights_only=True)
        load_adapter_state_dict(model, state)
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, device, dtype, injection, model_name


def tokenize_entities(entities: list[str], tokenizer) -> dict[str, list[int]]:
    mapping = {entity: list(tokenizer(entity, add_special_tokens=False)["input_ids"]) for entity in entities}
    empty = [entity for entity, tokens in mapping.items() if not tokens]
    if empty:
        raise ValueError(f"empty entity tokenization: {empty[:5]}")
    EntityTokenTrie(mapping)  # collision audit
    return mapping


def _memory(device) -> int:
    import torch
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def decode_free(model, tokenizer, rows: list[dict], protocol: str, *, max_new_tokens: int, batch_size: int, device) -> tuple[list[dict], dict]:
    import torch

    predictions = []
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [build_prompt(row, protocol) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        suffix = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(suffix, skip_special_tokens=True)
        for row, prompt, text in zip(batch, prompts, texts):
            predictions.append({**row, "prompt_protocol": protocol, "evaluation_prompt": prompt, "prediction_text": text.strip()})
    elapsed = time.perf_counter() - started
    return predictions, {"elapsed_seconds": elapsed, "latency_seconds_per_query": elapsed / max(1, len(rows)), "peak_gpu_memory_bytes": _memory(device)}


def decode_trie(model, tokenizer, rows: list[dict], protocol: str, trie: EntityTokenTrie, *, beam_size: int, device) -> tuple[list[dict], dict]:
    import torch

    predictions = []
    started = time.perf_counter()
    max_entity_tokens = max(len(tokens) for tokens in trie.token_sequences)
    for row in rows:
        prompt = build_prompt(row, protocol)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])
        constraint = TrieConstraint(trie, prompt_length, tokenizer.eos_token_id)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_entity_tokens + 1,
                do_sample=False,
                num_beams=int(beam_size),
                num_return_sequences=1,
                prefix_allowed_tokens_fn=constraint,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )
        token_ids = generated[0, prompt_length:].tolist()
        if tokenizer.eos_token_id in token_ids:
            token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
        try:
            entity = trie.entity_for(token_ids)
        except KeyError:
            entity = ""
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        predictions.append({**row, "prompt_protocol": protocol, "evaluation_prompt": prompt, "prediction_text": text, "canonical_prediction": entity, "generated_token_ids": token_ids})
    elapsed = time.perf_counter() - started
    return predictions, {"elapsed_seconds": elapsed, "latency_seconds_per_query": elapsed / max(1, len(rows)), "peak_gpu_memory_bytes": _memory(device), "beam_size": int(beam_size)}

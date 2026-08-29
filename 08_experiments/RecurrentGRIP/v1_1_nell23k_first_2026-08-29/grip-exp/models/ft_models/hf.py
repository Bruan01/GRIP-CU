from typing import Optional

from transformers import PreTrainedTokenizer, PreTrainedModel

from constants import TORCH_DTYPE, HF_DECODER_ONLY_LLMS, MODELSCOPE_DECODER_ONLY_LLMS
from models.utils import get_hf_llm_tokenizer, get_hf_tokenizer


def get_hf_ft_model(
        model_name: str,
        load_dir: Optional[str] = None,
        model_source: str = "auto",
        model_cache_dir: Optional[str] = "model_cache",
        local_files_only: bool = False,
        model_download_workers: int = 1,
        padding_side: str = "left",
        truncation_side: str = "left",
        tokenize_max_length: int = 4096,
        quantization: bool = False,
        dtype: str = "bfloat16",
        use_prefix: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.0,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128,
        **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    return get_hf_llm_tokenizer(
        model_name=(MODELSCOPE_DECODER_ONLY_LLMS[model_name]
                    if model_source in {"auto", "modelscope"}
                    else HF_DECODER_ONLY_LLMS[model_name]),
        load_dir=load_dir,
        model_source=model_source,
        model_cache_dir=model_cache_dir,
        local_files_only=local_files_only,
        model_download_workers=model_download_workers,
        padding_side=padding_side,
        truncation_side=truncation_side,
        tokenize_max_length=tokenize_max_length,
        use_fast=True,
        flash_attention=False,
        dtype=TORCH_DTYPE[dtype],
        quantization=quantization,
        peft=True,
        use_prefix=use_prefix,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        dropout=dropout,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
        **kwargs,
    )


def get_hf_ft_tokenizer(
        model_name: str,
        model_source: str = "auto",
        model_cache_dir: Optional[str] = "model_cache",
        local_files_only: bool = False,
        model_download_workers: int = 1,
        padding_side: str = "left",
        truncation_side: str = "left",
        tokenize_max_length: int = 4096,
        **kwargs,
) -> PreTrainedTokenizer:
    """Load the fine-tuning tokenizer without placing the base model on GPU."""
    model_id = (MODELSCOPE_DECODER_ONLY_LLMS[model_name]
                if model_source in {"auto", "modelscope"}
                else HF_DECODER_ONLY_LLMS[model_name])
    return get_hf_tokenizer(
        model_name=model_id,
        model_source=model_source,
        model_cache_dir=model_cache_dir,
        local_files_only=local_files_only,
        model_download_workers=model_download_workers,
        padding_side=padding_side,
        truncation_side=truncation_side,
        tokenize_max_length=tokenize_max_length,
        use_fast=True,
        **kwargs,
    )

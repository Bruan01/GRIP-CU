from typing import Optional

from .base import BaseInferenceModel
from .claude import ChatClaude
from .hf import HuggingFaceLLMs
from .oai import ChatOpenAI


def get_inf_model(
        model_name: str,
        load_dir: Optional[str] = None,
        quantization: bool = False,
        padding_side: str = "left",
        truncation_side: str = "right",
        tokenize_max_length: int = 20000,
        dtype: str = "bfloat16",
        flash_attention: bool = False,
        use_vllm: bool = True,
        gen_max_length: int = 10000,
        batch_size: int = -1,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 0.6,
        api_key: str = "${OPENAI_API_KEY}",
        api_base: str = "https://api.openai.com/v1/",
        api_version: Optional[str] = None,
        **kwargs,
) -> BaseInferenceModel:
    if model_name.startswith("claude"):
        return ChatClaude(
            api_key=api_key,
            model=model_name,
            batch_size=batch_size,
            gen_max_length=gen_max_length,
            **kwargs, )
    elif model_name.startswith("gpt"):
        return ChatOpenAI(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            model=model_name,
            batch_size=batch_size,
            gen_max_length=gen_max_length,
            **kwargs,
        )
    else:
        return HuggingFaceLLMs(
            model_name=model_name,
            load_dir=load_dir,
            quantization=quantization,
            padding_side=padding_side,
            truncation_side=truncation_side,
            tokenize_max_length=tokenize_max_length,
            dtype=dtype,
            flash_attention=flash_attention,
            use_vllm=use_vllm,
            batch_size=batch_size,
            gen_max_length=gen_max_length,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            **kwargs,
        )

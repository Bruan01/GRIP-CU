from datetime import datetime
from typing import Union, Optional

import torch
from tqdm.autonotebook import trange
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
try:
    from vllm import LLM, SamplingParams
except ImportError:  # vLLM is optional when using the Hugging Face generation path.
    LLM = None
    SamplingParams = None

from constants import HF_DECODER_ONLY_LLMS, MODELSCOPE_DECODER_ONLY_LLMS, SYSTEM_PROMPT, TORCH_DTYPE
from models.utils import get_hf_llm_tokenizer
from .base import BaseInferenceModel


class HuggingFaceLLMs(BaseInferenceModel):
    SUPPORTED_MODELS = HF_DECODER_ONLY_LLMS

    def __init__(self,
                 model_name: str,
                 load_dir: str = None,
                 quantization: bool = False,
                 padding_side: str = "left",
                 truncation_side: str = "right",
                 tokenize_max_length: int = 4096,
                 dtype: str = "bfloat16",
                 batch_size: int = -1,
                 gen_max_length: int = 10000,
                 do_sample: bool = True,
                 top_p: float = 0.9,
                 temperature: float = 0.6,
                 use_vllm: bool = True,
                 flash_attention: bool = False,
                 model_source: str = "auto",
                 model_cache_dir: str = "model_cache",
                 local_files_only: bool = False,
                 model_download_workers: int = 1,
                 **kwargs,
                 ):
        super().__init__()
        self.model_name = model_name
        self.load_dir = load_dir
        self.quantization = quantization
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.tokenize_max_length = tokenize_max_length
        self.dtype = dtype
        self.use_vllm = use_vllm
        self.flash_attention = flash_attention
        self.model_source = model_source
        self.model_cache_dir = model_cache_dir
        self.local_files_only = local_files_only
        self.model_download_workers = model_download_workers
        self.batch_size = batch_size
        self.gen_max_length = gen_max_length
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.kwargs = kwargs
        self._batch_inference = None

        # initialize the llm.
        self._get_llm_model()

    def _create_bnb_config(self):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_use_double_quant=False,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=TORCH_DTYPE[self.dtype])

        return bnb_config

    def _get_llm_model(self):
        if self.use_vllm:
            self._get_vllm_model()
        else:
            self._get_hf_llm()

    def _get_vllm_model(self):
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM is not installed. Use --use_vllm False for Hugging Face inference.")
        self._batch_inference = self.vllm_batch_inference
        if self.load_dir is not None:
            llm_model = self.load_dir
            model_source = "local"
        else:
            llm_model = (MODELSCOPE_DECODER_ONLY_LLMS[self.model_name]
                         if self.model_source in {"auto", "modelscope"}
                         else self.SUPPORTED_MODELS[self.model_name])
            model_source = self.model_source

        from models.utils import resolve_model_path
        llm_model = resolve_model_path(
            llm_model,
            model_source=model_source,
            model_cache_dir=self.model_cache_dir,
            local_files_only=self.local_files_only,
            model_download_workers=self.model_download_workers,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            use_fast=True,
            padding_side=self.padding_side,
            truncation_side=self.truncation_side,
            local_files_only=self.local_files_only,
        )

        if not self.do_sample:
            self.sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=self.gen_max_length,
            )
        else:
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.gen_max_length,
                truncate_prompt_tokens=self.tokenize_max_length,
            )
        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size % 2 != 0 and tensor_parallel_size != 1:
            tensor_parallel_size = tensor_parallel_size - 1
            print(f"Cannot use odd number of GPU for VLLM inference with number larger than 1, "
                  f"automatically use {tensor_parallel_size} GPUs.")

        if self.quantization:
            self.model = LLM(
                llm_model,
                tensor_parallel_size=tensor_parallel_size,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
                max_model_len=self.tokenize_max_length + self.gen_max_length,
                dtype=TORCH_DTYPE[self.dtype], )
        else:
            self.model = LLM(
                llm_model,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=self.tokenize_max_length + self.gen_max_length,
                dtype=TORCH_DTYPE[self.dtype], )

    def _get_hf_llm(self):
        self._batch_inference = self.hf_batch_inference

        self.model, self.tokenizer = get_hf_llm_tokenizer(
            model_name=(MODELSCOPE_DECODER_ONLY_LLMS[self.model_name]
                        if self.model_source in {"auto", "modelscope"}
                        else self.SUPPORTED_MODELS[self.model_name]),
            load_dir=self.load_dir,
            model_source=self.model_source,
            model_cache_dir=self.model_cache_dir,
            local_files_only=self.local_files_only,
            model_download_workers=self.model_download_workers,
            device_map="auto",
            padding_side=self.padding_side,
            truncation_side=self.truncation_side,
            tokenize_max_length=self.tokenize_max_length,
            use_fast=True,
            flash_attention=self.flash_attention,
            bnb_config=self._create_bnb_config() if self.quantization else None,
            dtype=TORCH_DTYPE[self.dtype],
        )

    def create_single_input(
            self,
            user_content: str,
            system_prompt: Optional[str] = None,
            history: list = None) -> str:
        if history is not None:
            conversation = history + [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        else:
            system_prompt = {
                "role": "system",
                "content": system_prompt or SYSTEM_PROMPT,
            }

            user_prompt = {
                "role": "user",
                "content": user_content
            }
            conversation = [system_prompt, user_prompt]
        single_input = self.tokenizer.apply_chat_template(conversation,
                                                          tokenize=False,
                                                          add_generation_prompt=True,
                                                          )
        return single_input

    def output_post_process(self, output_text: str) -> str:
        return output_text

    def hf_batch_inference(
            self,
            batch_input: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None) -> list[dict]:
        if histories is not None:
            batch_prompt_input = [self.create_single_input(input_text, system_prompt, history)
                                  for input_text, history in zip(batch_input, histories)]
        else:
            batch_prompt_input = [self.create_single_input(input_text, system_prompt)
                                  for input_text in batch_input]
        inputs = self.tokenizer(batch_prompt_input, return_tensors="pt", padding=True,
                                max_length=self.tokenize_max_length,
                                truncation=True)
        input_device = next(self.model.parameters()).device
        inputs = inputs.to(input_device)
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.gen_max_length,
            eos_token_id=terminators,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        response = outputs[:, inputs["input_ids"].shape[-1]:]
        response_texts = self.tokenizer.batch_decode(response, skip_special_tokens=True)
        cleaned_responses = []
        for query, text in zip(batch_input, response_texts):
            cleaned_responses.append({
                "query": query,
                "response": self.output_post_process(text),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
        return cleaned_responses

    def batch_inference(
            self,
            batch_input: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None) -> list[dict]:
        return self._batch_inference(batch_input, system_prompt, histories)

    def vllm_batch_inference(
            self,
            batch_input: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None
    ) -> list[dict]:
        if histories is not None:
            batch_prompt_input = [self.create_single_input(input_text, system_prompt, history)
                                  for input_text, history in zip(batch_input, histories)]
        else:
            batch_prompt_input = [self.create_single_input(input_text, system_prompt) for input_text in batch_input]
        outputs = self.model.generate(batch_prompt_input, sampling_params=self.sampling_params)
        cleaned_responses = []
        for query, output in zip(batch_input, outputs):
            cleaned_responses.append({
                "query": query,
                "response": self.output_post_process(output.outputs[0].text),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
        return cleaned_responses

    def inference(
            self,
            user_contents: list[Union[str, dict]],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None
    ) -> list[dict]:
        if isinstance(user_contents[0], dict):
            user_contents = [user_content["query"] for user_content in user_contents]
        if self.batch_size == -1:
            return self.batch_inference(user_contents, system_prompt, histories)
        else:
            results = []
            for start_index in trange(0, len(user_contents), self.batch_size, desc=f"Batch",
                                      disable=False, ):
                batch_contents = user_contents[start_index: start_index + self.batch_size]
                batch_histories = histories[
                                  start_index: start_index + self.batch_size] if histories is not None else None
                batch_result = self.batch_inference(batch_contents, system_prompt, batch_histories)
                results.extend(batch_result)
            return results

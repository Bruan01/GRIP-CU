import hashlib
import json
import random
from pathlib import Path
from abc import abstractmethod, ABC
from typing import Optional

from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from constants import SYSTEM_PROMPT
from models import BaseInferenceModel
from .train_system_prompts import TRAIN_SYSTEM_PROMPTS


class GenGraphTaskBase(ABC):
    task_system_prompts = TRAIN_SYSTEM_PROMPTS

    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator: Optional[BaseInferenceModel] = None,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            task_cache_dir: Optional[str] = None,
            **kwargs,
    ):
        self.graph_list = graph_list
        self.title_list = title_list
        self.tokenizer = tokenizer
        self.task_generator_model_name = task_generator_model_name
        self.task_gen_max_length = task_gen_max_length
        self.task_generator = task_generator
        self.task_cache_dir = Path(task_cache_dir) if task_cache_dir else None
        self.kwargs = kwargs
        super().__init__()

    @abstractmethod
    def gen_task(self, gen_empty_task=False) -> list:
        """
        generate train task.
        """
        pass

    def create_chat_message(self, question, answer):

        message = [
            {
                "role": "system",
                "content": random.sample(self.task_system_prompts, 1)[0]
                ,
            },
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
            }
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False)

    def __call__(self, gen_empty_task=False) -> list:
        return self.gen_task(gen_empty_task)

    def infer_cached(self, stage: str, prompts: list[str], system_prompt: str) -> list[dict]:
        """Generate task text with an append-only prompt-indexed JSONL cache.

        Long paper-scale task generation can outlive a job time limit.  Every
        completed batch is flushed to disk so rerunning the same seeded command
        reuses it instead of starting from zero.  The prompt digest prevents a
        stale cache record from being used after a prompt/config change.
        """
        if not prompts:
            return []
        if self.task_generator is None:
            raise RuntimeError(f"Task generator is required for stage {stage}.")
        if self.task_cache_dir is None:
            return self.task_generator.inference(prompts, system_prompt)

        self.task_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.task_cache_dir / f"{stage}.jsonl"
        cached: dict[tuple[int, str], dict] = {}
        if cache_path.is_file():
            with cache_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if isinstance(row.get("response"), dict):
                            cached[(row["index"], row["prompt_sha256"])] = row["response"]
                    except (KeyError, TypeError, json.JSONDecodeError):
                        continue

        digests = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
        results: list[Optional[dict]] = [cached.get((index, digest)) for index, digest in enumerate(digests)]
        missing = [index for index, result in enumerate(results) if result is None]
        if not missing:
            print(f"Task cache hit: {stage} ({len(prompts)} prompts)")
            return results

        batch_size = max(1, int(getattr(self.task_generator, "batch_size", 1)))
        cached_count = len(prompts) - len(missing)
        print(f"Task cache: {stage}; reusing {cached_count}/{len(prompts)}, generating {len(missing)}")
        with cache_path.open("a", encoding="utf-8") as stream:
            progress = tqdm(
                total=len(prompts),
                initial=cached_count,
                desc=f"Task generation [{stage}]",
                unit="prompt",
                dynamic_ncols=True,
            )
            try:
                for start in range(0, len(missing), batch_size):
                    indices = missing[start:start + batch_size]
                    batch_prompts = [prompts[index] for index in indices]
                    batch_results = self.task_generator.inference(batch_prompts, system_prompt)
                    if len(batch_results) != len(indices):
                        raise RuntimeError(f"Task generator returned {len(batch_results)} results for {len(indices)} prompts in {stage}.")
                    for index, response in zip(indices, batch_results):
                        results[index] = response
                        stream.write(json.dumps({
                            "index": index,
                            "prompt_sha256": digests[index],
                            "response": response,
                        }, ensure_ascii=False) + "\n")
                    stream.flush()
                    progress.update(len(indices))
            finally:
                progress.close()

        return results


    def sample_post_process(self, text_list: list):
        end = self.tokenizer.eos_token + "\n"
        if text_list and isinstance(text_list[0], list):
            return [self.sample_post_process(text) for text in text_list]
        else:
            return_list = []
            for text in text_list:
                # if no end of token is appended, manually append it.
                if text[-len(end):] != end:
                    text += end
                return_list.append(text)
            return return_list
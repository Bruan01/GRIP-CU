import hashlib
import json
import os
import random
import time
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
        # The prompt digest is the stable identity.  Including the position in
        # the key makes resume unsafe when an earlier completed graph is
        # removed from a pending subset and all later prompt indices shift.
        cached: dict[str, dict] = {}
        if cache_path.is_file():
            with cache_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if isinstance(row.get("response"), dict):
                            cached[row["prompt_sha256"]] = row["response"]
                    except (KeyError, TypeError, json.JSONDecodeError):
                        continue

        digests = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
        results: list[Optional[dict]] = [cached.get(digest) for digest in digests]
        missing = [index for index, result in enumerate(results) if result is None]
        if not missing:
            print(f"Task cache hit: {stage} ({len(prompts)} prompts)")
            self._write_progress(
                stage,
                total=len(prompts),
                completed=len(prompts),
                cached=len(prompts),
                generated=0,
                started_at=time.monotonic(),
                status="completed",
            )
            return results

        batch_size = max(1, int(getattr(self.task_generator, "batch_size", 1)))
        cached_count = len(prompts) - len(missing)
        print(f"Task cache: {stage}; reusing {cached_count}/{len(prompts)}, generating {len(missing)}")
        started_at = time.monotonic()
        completed_count = cached_count
        self._write_progress(
            stage,
            total=len(prompts),
            completed=completed_count,
            cached=cached_count,
            generated=0,
            started_at=started_at,
            status="running",
        )
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
                    failed = [
                        (index, response)
                        for index, response in zip(indices, batch_results)
                        if not isinstance(response, dict) or response.get("status") != "success"
                    ]
                    # Persist successful API calls even when another item in
                    # this concurrent batch failed. This avoids paying for
                    # already completed requests again after a rate-limit or
                    # transient error interrupts the batch.
                    successful = [
                        (index, response)
                        for index, response in zip(indices, batch_results)
                        if isinstance(response, dict) and response.get("status") == "success"
                    ]
                    for index, response in successful:
                        results[index] = response
                        stream.write(json.dumps({
                            "index": index,
                            "prompt_sha256": digests[index],
                            "response": response,
                        }, ensure_ascii=False) + "\n")
                    if successful:
                        stream.flush()
                        os.fsync(stream.fileno())
                        progress.update(len(successful))
                        completed_count += len(successful)
                        self._write_progress(
                            stage,
                            total=len(prompts),
                            completed=completed_count,
                            cached=cached_count,
                            generated=completed_count - cached_count,
                            started_at=started_at,
                            status="running",
                        )
                    if failed:
                        preview = []
                        for index, response in failed[:3]:
                            if isinstance(response, dict):
                                preview.append(f"index={index}: {response.get('error', 'unknown API error')}")
                            else:
                                preview.append(f"index={index}: invalid response type {type(response).__name__}")
                        raise RuntimeError(
                            f"Task generation batch failed for {stage}; failed responses were not cached. "
                            + "; ".join(preview)
                        )
                    if not successful:
                        continue
                    if completed_count == len(prompts):
                        self._write_progress(
                            stage,
                            total=len(prompts),
                            completed=completed_count,
                            cached=cached_count,
                            generated=completed_count - cached_count,
                            started_at=started_at,
                            status="completed",
                        )
            except Exception as exc:
                self._write_progress(
                    stage,
                    total=len(prompts),
                    completed=completed_count,
                    cached=cached_count,
                    generated=completed_count - cached_count,
                    started_at=started_at,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            else:
                self._write_progress(
                    stage,
                    total=len(prompts),
                    completed=completed_count,
                    cached=cached_count,
                    generated=completed_count - cached_count,
                    started_at=started_at,
                    status="completed",
                )
            finally:
                progress.close()

        return results

    def _write_progress(
            self,
            stage: str,
            total: int,
            completed: int,
            cached: int,
            generated: int,
            started_at: float,
            status: str,
            error: Optional[str] = None,
    ) -> None:
        """Write a durable, machine-readable progress snapshot for a stage."""
        if self.task_cache_dir is None:
            return
        elapsed = max(0.0, time.monotonic() - started_at)
        rate = generated / elapsed if elapsed > 0 and generated > 0 else 0.0
        remaining = max(0, total - completed)
        eta_seconds = remaining / rate if rate > 0 else None
        payload = {
            "stage": stage,
            "status": status,
            "total": total,
            "completed": completed,
            "cached": cached,
            "generated": generated,
            "percent": round(100.0 * completed / total, 4) if total else 100.0,
            "rate_prompt_per_second": round(rate, 4),
            "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if error:
            payload["error"] = error[:1000]
        progress_path = self.task_cache_dir / f"{stage}.progress.json"
        temp_path = progress_path.with_name(progress_path.name + ".tmp")
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, progress_path)


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

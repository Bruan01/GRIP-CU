"""D2 exact global entity scoring with one prompt forward and chunked KV-cache reuse."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

from .metrics import ranking_metrics
from .records import build_prompt


def _repeat_cache(cache, repeats: int):
    if isinstance(cache, tuple):
        return tuple(tuple(value.repeat_interleave(repeats, dim=0) for value in layer) for layer in cache)
    duplicated = copy.deepcopy(cache)
    if hasattr(duplicated, "batch_repeat_interleave"):
        duplicated.batch_repeat_interleave(repeats)
        return duplicated
    raise TypeError(f"unsupported past_key_values type: {type(cache)!r}")


@dataclass
class ScoreOutput:
    sum_scores: list[float]
    mean_scores: list[float]
    elapsed_seconds: float


class GlobalEntityScorer:
    def __init__(self, model, tokenizer, entities: list[str], entity_tokens: dict[str, list[int]], *, device, dtype, chunk_size: int = 256) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.entities = entities
        self.entity_tokens = entity_tokens
        self.device = device
        self.dtype = dtype
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def score_prompt(self, prompt: str) -> ScoreOutput:
        import torch
        import torch.nn.functional as F

        started = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"].to(self.device)
        with torch.inference_mode():
            prompt_output = self.model(input_ids=input_ids, attention_mask=attention, use_cache=True)
            first_log_probs = F.log_softmax(prompt_output.logits[:, -1, :].float(), dim=-1)[0]
        all_sum, all_mean = [], []
        prompt_length = input_ids.shape[1]
        for start in range(0, len(self.entities), self.chunk_size):
            batch_entities = self.entities[start : start + self.chunk_size]
            sequences = [self.entity_tokens[entity] for entity in batch_entities]
            batch_size = len(sequences)
            lengths = [len(sequence) for sequence in sequences]
            first_tokens = torch.tensor(
                [sequence[0] for sequence in sequences],
                dtype=torch.long,
                device=self.device,
            )
            scores = first_log_probs.index_select(0, first_tokens).clone()
            maximum_prefix = max(lengths) - 1
            if maximum_prefix > 0:
                pad_id = self.tokenizer.pad_token_id
                prefix_rows, prefix_masks = [], []
                for sequence in sequences:
                    prefix = sequence[:-1]
                    pad = maximum_prefix - len(prefix)
                    prefix_rows.append(prefix + [pad_id] * pad)
                    prefix_masks.append([1] * len(prefix) + [0] * pad)
                prefix_ids = torch.tensor(prefix_rows, dtype=torch.long, device=self.device)
                full_attention = torch.cat([
                    attention.repeat(batch_size, 1),
                    torch.tensor(prefix_masks, dtype=attention.dtype, device=self.device),
                ], dim=1)
                repeated_cache = _repeat_cache(prompt_output.past_key_values, batch_size)
                with torch.inference_mode():
                    output = self.model(input_ids=prefix_ids, attention_mask=full_attention, past_key_values=repeated_cache, use_cache=False)
                    continuation_log_probs = F.log_softmax(output.logits.float(), dim=-1)
                for row_index, sequence in enumerate(sequences):
                    for token_index in range(1, len(sequence)):
                        scores[row_index] += continuation_log_probs[row_index, token_index - 1, sequence[token_index]]
                del repeated_cache, output, continuation_log_probs
            values = scores.detach().cpu().tolist()
            all_sum.extend(values)
            all_mean.extend(value / length for value, length in zip(values, lengths))
        return ScoreOutput(sum_scores=all_sum, mean_scores=all_mean, elapsed_seconds=time.perf_counter() - started)


def rank_entities(entities: list[str], scores: list[float], answer: str, top_k: int = 10) -> dict:
    order = sorted(range(len(entities)), key=lambda index: (-scores[index], entities[index]))
    answer_index = entities.index(str(answer))
    rank = order.index(answer_index) + 1
    return {
        "rank": rank,
        "gold_score": scores[answer_index],
        "top_entities": [{"entity": entities[index], "score": scores[index]} for index in order[:top_k]],
        "prediction": entities[order[0]],
    }


def score_rows(scorer: GlobalEntityScorer, rows: list[dict], protocol: str, top_k: int = 10) -> tuple[list[dict], dict]:
    outputs = []
    total_elapsed = 0.0
    for row in rows:
        prompt = build_prompt(row, protocol)
        scores = scorer.score_prompt(prompt)
        total_elapsed += scores.elapsed_seconds
        ranked_sum = rank_entities(scorer.entities, scores.sum_scores, row["answer"], top_k)
        ranked_mean = rank_entities(scorer.entities, scores.mean_scores, row["answer"], top_k)
        outputs.append({
            **row,
            "prompt_protocol": protocol,
            "evaluation_prompt": prompt,
            "sum": ranked_sum,
            "mean": ranked_mean,
        })
    metrics = {
        "sum": ranking_metrics(row["sum"]["rank"] for row in outputs),
        "mean": ranking_metrics(row["mean"]["rank"] for row in outputs),
        "elapsed_seconds": total_elapsed,
        "latency_seconds_per_query": total_elapsed / max(1, len(outputs)),
        "entity_count": len(scorer.entities),
        "chunk_size": scorer.chunk_size,
    }
    return outputs, metrics


def select_normalization(validation_metrics: dict, candidates: list[str]) -> str:
    return max(candidates, key=lambda name: (validation_metrics[name]["hit_at_1"], validation_metrics[name]["mrr"], -candidates.index(name)))


def score_candidate_rows(model, tokenizer, rows: list[dict], protocol: str, candidate_builder, entity_tokens: dict[str, list[int]], *, device, dtype, chunk_size: int = 4, top_k: int = 4) -> tuple[list[dict], dict]:
    """D3 diagnostic: score gold plus train-only distractors; never a primary graph-free metric."""
    outputs = []
    total_elapsed = 0.0
    for row in rows:
        candidates = candidate_builder(row)
        scorer = GlobalEntityScorer(model, tokenizer, candidates, entity_tokens, device=device, dtype=dtype, chunk_size=chunk_size)
        prompt = build_prompt(row, protocol)
        scores = scorer.score_prompt(prompt)
        total_elapsed += scores.elapsed_seconds
        outputs.append({
            **row,
            "prompt_protocol": protocol,
            "evaluation_prompt": prompt,
            "candidate_source": "gold_plus_train_only_same_depth_distractors",
            "candidate_count": len(candidates),
            "sum": rank_entities(candidates, scores.sum_scores, row["answer"], top_k),
            "mean": rank_entities(candidates, scores.mean_scores, row["answer"], top_k),
        })
    metrics = {
        "sum": ranking_metrics(row["sum"]["rank"] for row in outputs),
        "mean": ranking_metrics(row["mean"]["rank"] for row in outputs),
        "elapsed_seconds": total_elapsed,
        "latency_seconds_per_query": total_elapsed / max(1, len(outputs)),
        "diagnostic_only": True,
    }
    return outputs, metrics

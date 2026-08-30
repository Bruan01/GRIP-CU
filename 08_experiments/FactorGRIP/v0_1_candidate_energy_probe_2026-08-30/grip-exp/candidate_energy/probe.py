"""Core candidate-energy probe orchestration.

Loads a RecurrentGRIP model with pre-trained v1.1.1 LoRA adapters and runs
three decoder types (free / constrained / score) under three adapter controls
(correct / none / wrong_depth) on the NELL23K validation and test splits.

The probe is inference-only: it never re-trains and never constructs an input
from the target relation.  Candidate scores are computed with one batched
forward pass per question; the ``score`` decoder is just the argmax over those
scores.  Normalisation (length-normalised mean vs raw sum) is selected on the
validation split and frozen before the test split is scored.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from transformers import LogitsProcessorList

from evaluation.recurrent_metrics import exact_match
from grip.recurrent import set_recurrent_depth
from grip.tasks.eval_tasks.grip_eval import GRIPEvalDataset

from .constrained import CandidateTrieLogitsProcessor, build_candidate_trie
from .scoring import score_candidates_batched

ADAPTER_CONTROLS = ["correct", "none", "wrong_depth"]
DECODER_TYPES = ["free", "constrained", "score"]
PROBE_EVAL_K = 1  # probe focuses on K=1 per README

_ADAPTER_NAME = {"correct": "eval_correct", "wrong_depth": "eval_wrong_depth"}
_ADAPTER_TRAIN_K = {"correct": 1, "none": 0, "wrong_depth": 2}

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _parse_answer(text: str) -> str:
    """Replicated from v1.1.1 runner (avoid importing from ``scripts/``)."""

    matches = _ANSWER_RE.findall(text)
    if matches:
        return "; ".join(item.strip() for item in matches).strip()
    stripped = text.strip()
    bracket = re.fullmatch(r"\[\s*(.*?)\s*\]", stripped, flags=re.DOTALL)
    return (bracket.group(1) if bracket else stripped).strip()


@dataclass
class CandidateProbePrediction:
    """One prediction for one (question, adapter_control, decoder_type) cell."""

    question_id: str
    split: str
    target: list[str]
    candidates: list[str]
    adapter_control: str
    decoder_type: str
    predicted_candidate: str | None
    raw_response: str | None
    response: str | None
    correct: bool
    latency_seconds: float
    peak_memory_bytes: int
    recurrent_train_k: int
    eval_k: int
    candidate_score_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.adapter_control not in {"correct", "none", "wrong_depth", "shuffled"}:
            raise ValueError(f"invalid adapter_control: {self.adapter_control}")
        if self.decoder_type not in {"free", "constrained", "score"}:
            raise ValueError(f"invalid decoder_type: {self.decoder_type}")
        if self.eval_k < 1:
            raise ValueError("eval_k must be positive")
        if self.latency_seconds < 0:
            raise ValueError("latency cannot be negative")


@dataclass
class CandidateScoreRecord:
    """Per-candidate scores for one (question, adapter_control) cell."""

    question_id: str
    split: str
    target: list[str]
    candidates: list[str]
    adapter_control: str
    recurrent_train_k: int
    eval_k: int
    latency_seconds: float
    scores: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@contextmanager
def _adapter_context(model, adapter_control: str) -> Iterator[None]:
    if adapter_control == "none":
        with model.disable_adapter():
            yield
        return
    model.set_adapter(_ADAPTER_NAME[adapter_control])
    yield


def _build_eval_dataset(record: dict, tokenizer) -> tuple[list[dict], GRIPEvalDataset]:
    samples = [
        item
        for item in record["recurrent_questions"]
        if item["split"] in {"validation", "test"}
    ]
    dataset = GRIPEvalDataset(
        questions=[item["question"] for item in samples],
        answers=[item["answer"] for item in samples],
        tokenizer=tokenizer,
        graph=record["graph"],
        title=record.get("title", record.get("id", "graph")),
        no_graph_context=True,
    )
    return samples, dataset


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, TypeError):
        return next(model.parameters()).device


def _peak_memory(device: torch.device) -> int:
    if torch.cuda.is_available() and device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def _reset_peak_memory(device: torch.device) -> None:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _run_free_decoder(
    model, tokenizer, input_ids, attention_mask, gen_max_length: int
) -> tuple[str, str]:
    """Greedy generation replicating v1.1.1 eval (returns parsed, raw)."""

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=gen_max_length,
            do_sample=False,
            use_cache=False,
        )
    new_ids = generated[0][input_ids.shape[-1] :]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    return _parse_answer(raw), raw.strip()


def _run_constrained_decoder(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    candidates: list[str],
    max_new_tokens: int,
) -> tuple[str, str]:
    """Trie-constrained greedy generation (returns parsed, raw)."""

    prompt_len = input_ids.shape[-1]
    trie = build_candidate_trie(tokenizer, candidates)
    vocab_size = model.config.vocab_size
    processor = CandidateTrieLogitsProcessor(
        trie=trie,
        prompt_length=prompt_len,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=vocab_size,
    )
    processors = LogitsProcessorList([processor])
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=processors,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
    new_ids = generated[0][input_ids.shape[-1] :]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    return _parse_answer(raw), raw.strip()


def _argmax_from_dicts(
    score_dicts: list[dict[str, Any]], normalized: bool
) -> tuple[int, float]:
    """Return (best_index, best_score) from a list of score dicts."""

    if not score_dicts:
        return -1, float("-inf")
    key = (lambda s: s["norm_logprob"]) if normalized else (lambda s: s["sum_logprob"])
    best_idx = max(range(len(score_dicts)), key=lambda i: key(score_dicts[i]))
    return best_idx, key(score_dicts[best_idx])


def _max_answer_tokens(tokenizer, candidates: list[str]) -> int:
    return (
        max(
            len(tokenizer.encode(f"<answer>{c}</answer>", add_special_tokens=False))
            for c in candidates
        )
        if candidates
        else 1
    )


def run_candidate_energy_probe(
    model,
    tokenizer,
    record: dict,
    adapter_paths: dict[str, str],
    device: torch.device,
    gen_max_length: int = 24,
    max_total_length: int = 1024,
    adapter_controls: list[str] | None = None,
    decoder_types: list[str] | None = None,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Run the full probe and return ``(predictions, candidate_scores, metadata)``.

    Parameters
    ----------
    adapter_paths:
        ``{"train_k1": <dir>, "train_k2": <dir>}`` — v1.1.1 adapter directories.
    """

    controls = adapter_controls or list(ADAPTER_CONTROLS)
    decoders = decoder_types or list(DECODER_TYPES)

    # ---- Load pre-trained adapters (inference-only) ----
    if "correct" in controls:
        model.load_adapter(
            adapter_paths["train_k1"], adapter_name="eval_correct", is_trainable=False
        )
    if "wrong_depth" in controls:
        model.load_adapter(
            adapter_paths["train_k2"],
            adapter_name="eval_wrong_depth",
            is_trainable=False,
        )

    set_recurrent_depth(model, PROBE_EVAL_K)
    model.config.use_cache = False
    model.eval()

    samples, dataset = _build_eval_dataset(record, tokenizer)
    constrained_max_new = max(
        gen_max_length,
        _max_answer_tokens(tokenizer, samples[0].get("candidate_relations", [])) + 4
        if samples
        else gen_max_length,
    )

    predictions: list[dict] = []
    candidate_scores: list[dict] = []
    # Map (question_id, control) -> score dict list, for score-decoder phase.
    score_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}

    # ---- Phase 1: scoring + free + constrained (needs model + adapter) ----
    for control in controls:
        with _adapter_context(model, control):
            train_k = _ADAPTER_TRAIN_K[control]
            for idx in range(len(dataset)):
                input_ids, _question, answer = dataset[idx]
                sample = samples[idx]
                target = [answer] if isinstance(answer, str) else list(answer)
                candidates = sample.get("candidate_relations", [])
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids)
                prompt_ids = input_ids[0].tolist()

                # --- Candidate scoring (one batched forward pass) ---
                _reset_peak_memory(device)
                t0 = time.perf_counter()
                scores = score_candidates_batched(
                    model,
                    tokenizer,
                    prompt_ids,
                    candidates,
                    device,
                    max_total_length=max_total_length,
                )
                score_latency = time.perf_counter() - t0
                score_peak = _peak_memory(device)
                score_dicts = [
                    {
                        "candidate": s.candidate,
                        "sum_logprob": s.sum_logprob,
                        "num_tokens": s.num_tokens,
                        "norm_logprob": s.norm_logprob,
                    }
                    for s in scores
                ]
                score_lookup[(sample["question_id"], control)] = score_dicts
                candidate_scores.append(
                    CandidateScoreRecord(
                        question_id=sample["question_id"],
                        split=sample["split"],
                        target=[str(t) for t in target],
                        candidates=candidates,
                        adapter_control=control,
                        recurrent_train_k=train_k,
                        eval_k=PROBE_EVAL_K,
                        latency_seconds=score_latency,
                        scores=score_dicts,
                    ).to_dict()
                )

                # --- Free decoder ---
                if "free" in decoders:
                    _reset_peak_memory(device)
                    t0 = time.perf_counter()
                    parsed, raw = _run_free_decoder(
                        model, tokenizer, input_ids, attention_mask, gen_max_length
                    )
                    latency = time.perf_counter() - t0
                    pred = CandidateProbePrediction(
                        question_id=sample["question_id"],
                        split=sample["split"],
                        target=[str(t) for t in target],
                        candidates=candidates,
                        adapter_control=control,
                        decoder_type="free",
                        predicted_candidate=parsed,
                        raw_response=raw,
                        response=parsed,
                        correct=exact_match(parsed, target),
                        latency_seconds=latency,
                        peak_memory_bytes=_peak_memory(device),
                        recurrent_train_k=train_k,
                        eval_k=PROBE_EVAL_K,
                    )
                    pred.validate()
                    predictions.append(pred.to_dict())

                # --- Constrained decoder ---
                if "constrained" in decoders:
                    _reset_peak_memory(device)
                    t0 = time.perf_counter()
                    parsed, raw = _run_constrained_decoder(
                        model,
                        tokenizer,
                        input_ids,
                        attention_mask,
                        candidates,
                        constrained_max_new,
                    )
                    latency = time.perf_counter() - t0
                    pred = CandidateProbePrediction(
                        question_id=sample["question_id"],
                        split=sample["split"],
                        target=[str(t) for t in target],
                        candidates=candidates,
                        adapter_control=control,
                        decoder_type="constrained",
                        predicted_candidate=parsed,
                        raw_response=raw,
                        response=parsed,
                        correct=exact_match(parsed, target),
                        latency_seconds=latency,
                        peak_memory_bytes=_peak_memory(device),
                        recurrent_train_k=train_k,
                        eval_k=PROBE_EVAL_K,
                    )
                    pred.validate()
                    predictions.append(pred.to_dict())

    # ---- Phase 2: normalization selection on validation ----
    best_norm = True
    best_val_acc = -1.0
    for norm in (True, False):
        correct_count = 0
        total = 0
        for cs in candidate_scores:
            if cs["split"] != "validation":
                continue
            best_idx, _ = _argmax_from_dicts(cs["scores"], norm)
            predicted = cs["candidates"][best_idx] if best_idx >= 0 else ""
            if exact_match(predicted, cs["target"]):
                correct_count += 1
            total += 1
        acc = correct_count / total if total else 0.0
        if acc > best_val_acc:
            best_val_acc = acc
            best_norm = norm

    # ---- Phase 3: score-decoder predictions with frozen normalization ----
    for cs in candidate_scores:
        best_idx, best_score = _argmax_from_dicts(cs["scores"], best_norm)
        predicted = cs["candidates"][best_idx] if best_idx >= 0 else ""
        pred = CandidateProbePrediction(
            question_id=cs["question_id"],
            split=cs["split"],
            target=cs["target"],
            candidates=cs["candidates"],
            adapter_control=cs["adapter_control"],
            decoder_type="score",
            predicted_candidate=predicted,
            raw_response=None,
            response=predicted,
            correct=exact_match(predicted, cs["target"]),
            latency_seconds=cs["latency_seconds"],
            peak_memory_bytes=0,
            recurrent_train_k=cs["recurrent_train_k"],
            eval_k=cs["eval_k"],
            candidate_score_summary={
                "normalization": "length_normalized" if best_norm else "raw_sum",
                "best_index": best_idx,
                "best_score": best_score,
            },
        )
        pred.validate()
        predictions.append(pred.to_dict())

    metadata = {
        "eval_k": PROBE_EVAL_K,
        "selected_normalization": "length_normalized" if best_norm else "raw_sum",
        "validation_normalization_accuracy": best_val_acc,
        "adapter_controls": controls,
        "decoder_types": decoders,
        "num_questions": len(dataset),
        "shuffled_control": "unavailable (single graph; requires >=2 trained graphs)",
    }
    return predictions, candidate_scores, metadata

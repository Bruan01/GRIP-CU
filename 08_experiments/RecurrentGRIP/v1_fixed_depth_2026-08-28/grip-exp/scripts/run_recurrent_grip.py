from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm
from transformers import TrainingArguments

from arguments import (
    BaseArguments,
    CustomTrainingArguments,
    InferenceArguments,
    ModelArguments,
    RecurrentArguments,
    TaskArguments,
    parse_args,
)
from evaluation.recurrent_metrics import exact_match
from grip.recurrent import (
    RecurrentPrediction,
    build_recurrent_peft_model,
    capture_recurrent_trace,
    set_recurrent_depth,
)
from grip.tasks.eval_tasks.grip_eval import GRIPEvalDataset
from grip.tasks.recurrent_tasks import build_recurrent_task_dataset
from grip.tasks.train_tasks.gen_context import GenGraphContextTask
from grip.training.recurrent_trainer import train_recurrent_graph
from utils import load_list_json, log_event, log_runtime, save_entry_to_list_json, set_random_seed


def _safe_id(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _graph_id(record: dict, graph_index: int) -> str:
    return str(record.get("id", record.get("title", graph_index)))


def _parse_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return "; ".join(item.strip() for item in matches).strip()


def _eval_dataset(record: dict, tokenizer):
    samples = [item for item in record["recurrent_questions"] if item["split"] in {"validation", "test"}]
    dataset = GRIPEvalDataset(
        questions=[item["question"] for item in samples],
        answers=[item["answer"] for item in samples],
        tokenizer=tokenizer,
        graph=record["graph"],
        title=record.get("title", _graph_id(record, 0)),
        no_graph_context=True,
    )
    return samples, dataset


@contextmanager
def _adapter_context(model, adapter_control: str) -> Iterator[None]:
    if adapter_control == "none":
        with model.disable_adapter():
            yield
        return
    model.set_adapter(f"eval_{adapter_control}")
    yield


def _requested_controls(adapter_control: str, num_graphs: int) -> list[str]:
    controls = ["correct", "shuffled", "none"] if adapter_control == "all" else [adapter_control]
    if num_graphs < 2 and "shuffled" in controls:
        if adapter_control == "shuffled":
            raise ValueError("The shuffled adapter control requires at least two trained graphs")
        controls.remove("shuffled")
        log_event("recurrent_shuffled_control_skipped", reason="requires_at_least_two_graphs")
    return controls


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, TypeError):
        return next(model.parameters()).device


def _resolve_evaluation_device(requested: str, rank: int = 0) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("evaluation_device=cuda was requested, but CUDA is unavailable")
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("evaluation_device=mps was requested, but MPS is unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def _place_model_for_evaluation(model, requested: str, rank: int = 0) -> torch.device:
    device = _resolve_evaluation_device(requested, rank=rank)
    model.to(device)
    log_event("recurrent_evaluation_model_placed", device=str(device))
    return device


def evaluate_depth_sweep(
    model,
    tokenizer,
    record: dict,
    exp_args: dict,
    output_file: str,
    adapter_ids: dict[str, str],
    controls: list[str],
    graph_index: int,
    run_started: float,
) -> None:
    samples, dataset = _eval_dataset(record, tokenizer)
    graph_id = _graph_id(record, graph_index)
    for adapter_control in controls:
        with _adapter_context(model, adapter_control):
            for depth in exp_args["recurrent_depth_sweep"]:
                set_recurrent_depth(model, depth)
                model.config.use_cache = False
                model.eval()
                for index in tqdm(
                    range(len(dataset)),
                    desc=f"eval graph={graph_id} adapter={adapter_control} K={depth}",
                ):
                    elapsed, _, hard_stop = _time_state(run_started, exp_args)
                    if hard_stop:
                        raise TimeoutError(
                            f"RecurrentGRIP hard stop reached after {elapsed / 60:.1f} minutes"
                        )
                    input_ids, question, answer = dataset[index]
                    target = [answer] if isinstance(answer, str) else list(answer)
                    device = _input_device(model)
                    input_ids = input_ids.to(device)
                    attention_mask = torch.ones_like(input_ids)
                    if torch.cuda.is_available() and device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    trace = []
                    if exp_args["save_step_hidden_states"]:
                        trace = capture_recurrent_trace(model, input_ids, attention_mask)
                    started = time.perf_counter()
                    with torch.no_grad():
                        generated = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=exp_args["gen_max_length"],
                            do_sample=False,
                            use_cache=False,
                        )
                    latency = time.perf_counter() - started
                    raw_response = tokenizer.decode(
                        generated[0][input_ids.shape[-1]:], skip_special_tokens=True
                    )
                    parsed = _parse_answer(raw_response)
                    peak_memory = (
                        torch.cuda.max_memory_allocated(device)
                        if torch.cuda.is_available() and device.type == "cuda"
                        else 0
                    )
                    sample = samples[index]
                    prediction = RecurrentPrediction(
                        graph_id=graph_id,
                        question_id=sample["question_id"],
                        question=str(question),
                        target=[str(value) for value in target],
                        true_hop=int(sample["true_hop"]),
                        recurrence_k=int(depth),
                        adapter_id=adapter_ids[adapter_control],
                        adapter_control=adapter_control,
                        raw_response=raw_response.strip(),
                        response=parsed,
                        correct=exact_match(parsed, target),
                        latency_seconds=latency,
                        peak_memory_bytes=int(peak_memory),
                        step_pooled_hidden_states=trace,
                        metadata={
                            "split": sample["split"],
                            "question_type": sample["question_type"],
                            "source_node": sample["source_node"],
                            "target_node": sample["target_node"],
                            "shortest_path": sample["shortest_path"],
                        },
                    )
                    prediction.validate()
                    save_entry_to_list_json(output_file, prediction.to_dict())


def _time_state(started: float, exp_args: dict) -> tuple[float, bool, bool]:
    elapsed = time.monotonic() - started
    return (
        elapsed,
        elapsed >= exp_args["wall_time_limit_minutes"] * 60,
        elapsed >= exp_args["hard_stop_minutes"] * 60,
    )


def _train_adapters(
    rank: int,
    records: list[dict],
    exp_args: dict,
    training_args: TrainingArguments,
    started: float,
) -> tuple[list[dict], dict[str, Path]]:
    set_random_seed(training_args.seed + rank)
    trained_records: list[dict] = []
    adapter_paths: dict[str, Path] = {}
    for graph_index, record in enumerate(records):
        elapsed, soft_stop, hard_stop = _time_state(started, exp_args)
        if hard_stop:
            raise TimeoutError(f"RecurrentGRIP hard stop reached after {elapsed / 60:.1f} minutes")
        if soft_stop:
            log_event("recurrent_soft_time_limit", phase="train", graph_index=graph_index, elapsed_seconds=elapsed)
            break

        graph_id = _graph_id(record, graph_index)
        safe_graph_id = _safe_id(graph_id)
        log_event("recurrent_graph_training_start", graph_index=graph_index, graph_id=graph_id)
        model, tokenizer, executor_index = build_recurrent_peft_model(**exp_args)
        context_samples = GenGraphContextTask(
            graph_list=[record["graph"]],
            title_list=[record.get("title", graph_id)],
            tokenizer=tokenizer,
            context_upsampling=exp_args["context_upsampling"],
            format_as_instruction=exp_args["format_as_instruction"],
        )()[0]
        task_dataset = build_recurrent_task_dataset(
            tokenizer=tokenizer,
            title=record.get("title", graph_id),
            context_samples=context_samples,
            samples=record["recurrent_questions"],
        )
        trainer_kwargs = dict(exp_args)
        trainer_kwargs.pop("executor_layer_index", None)
        trainer_kwargs.pop("recurrent_depth_train", None)
        model, tokenizer = train_recurrent_graph(
            model=model,
            tokenizer=tokenizer,
            training_dataset=task_dataset,
            training_args=training_args,
            executor_layer_index=executor_index,
            recurrent_depth_train=exp_args["recurrent_depth_train"],
            **trainer_kwargs,
        )
        adapter_dir = Path(exp_args["adapter_output_dir"]) / safe_graph_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        manifest = {
            "graph_id": graph_id,
            "safe_graph_id": safe_graph_id,
            "executor_layer_index": executor_index,
            "recurrent_depth_train": exp_args["recurrent_depth_train"],
            "target_modules": exp_args["target_modules"],
            "lora_r": exp_args["lora_r"],
            "lora_alpha": exp_args["lora_alpha"],
        }
        (adapter_dir / "recurrent_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        trained_records.append(record)
        adapter_paths[graph_id] = adapter_dir
        log_event(
            "recurrent_graph_training_complete",
            graph_index=graph_index,
            graph_id=graph_id,
            adapter_dir=str(adapter_dir.resolve()),
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return trained_records, adapter_paths


def _load_evaluation_adapters(
    model,
    graph_id: str,
    shuffled_graph_id: str,
    adapter_paths: dict[str, Path],
    controls: list[str],
) -> dict[str, str]:
    adapter_ids = {"none": "none"}
    if "correct" in controls:
        model.load_adapter(
            str(adapter_paths[graph_id]),
            adapter_name="eval_correct",
            is_trainable=False,
        )
        adapter_ids["correct"] = graph_id
    if "shuffled" in controls:
        model.load_adapter(
            str(adapter_paths[shuffled_graph_id]),
            adapter_name="eval_shuffled",
            is_trainable=False,
        )
        adapter_ids["shuffled"] = shuffled_graph_id
    return adapter_ids


def run(rank: int, records: list[dict], exp_args: dict, training_args: TrainingArguments, output_file: str) -> None:
    started = time.monotonic()
    trained_records, adapter_paths = _train_adapters(rank, records, exp_args, training_args, started)
    if not trained_records:
        raise RuntimeError("No graph adapter was trained before the wall-time limit")

    controls = _requested_controls(exp_args["adapter_control"], len(trained_records))
    graph_ids = [_graph_id(record, index) for index, record in enumerate(trained_records)]
    for graph_index, record in enumerate(trained_records):
        elapsed, _, hard_stop = _time_state(started, exp_args)
        if hard_stop:
            raise TimeoutError(f"RecurrentGRIP hard stop reached after {elapsed / 60:.1f} minutes")
        graph_id = graph_ids[graph_index]
        shuffled_graph_id = graph_ids[(graph_index + 1) % len(graph_ids)]
        model, tokenizer, _ = build_recurrent_peft_model(adapter_name="scratch", **exp_args)
        adapter_ids = _load_evaluation_adapters(
            model=model,
            graph_id=graph_id,
            shuffled_graph_id=shuffled_graph_id,
            adapter_paths=adapter_paths,
            controls=controls,
        )
        _place_model_for_evaluation(
            model,
            requested=exp_args.get("evaluation_device", "auto"),
            rank=rank,
        )
        evaluate_depth_sweep(
            model=model,
            tokenizer=tokenizer,
            record=record,
            exp_args=exp_args,
            output_file=output_file,
            adapter_ids=adapter_ids,
            controls=controls,
            graph_index=graph_index,
            run_started=started,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    log_runtime("recurrent_grip_runtime_environment")
    base_args, training_args, exp_args = parse_args(
        (
            BaseArguments,
            TrainingArguments,
            (TaskArguments, ModelArguments, CustomTrainingArguments, InferenceArguments, RecurrentArguments),
        ),
        no_dict=(TrainingArguments,),
    )
    rank = base_args.pop("rank")
    input_file = base_args.pop("input_file")
    base_args.pop("ref_file")
    output_file = base_args.pop("output_file")
    overwrite = base_args.pop("overwrite")
    exp_args.update(base_args)
    if exp_args.get("require_cuda") and not torch.cuda.is_available():
        raise RuntimeError("--require_cuda was enabled, but torch.cuda.is_available() is false")
    if exp_args["involve_qa_epochs"] < 1:
        raise ValueError("RecurrentGRIP requires --involve_qa_epochs >= 1 to train the 1-2 hop QA program")
    os.environ["RANK"] = str(rank)
    records = load_list_json(input_file)
    if exp_args["max_graphs"] > 0:
        records = records[: exp_args["max_graphs"]]
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()
    if not records:
        raise ValueError("No recurrent CLEGR records were loaded")
    run(rank, records, exp_args, training_args, output_file)


if __name__ == "__main__":
    main()

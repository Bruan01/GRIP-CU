import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm, trange
from transformers import TrainingArguments

from arguments import (
    BaseArguments,
    TaskArguments,
    ModelArguments,
    CustomTrainingArguments,
    InferenceArguments,
    parse_args
)
from grip.tasks import GripTaskGeneration, gen_eval_task
from grip.training import train
from models import get_ft_model, get_hf_ft_tokenizer
from utils import (extract_tag_content, load_list_json,
                   set_random_seed, Timer, log_event, log_runtime, model_summary, log_cuda_memory)


def _graph_id(record: dict) -> str:
    """Return the stable graph identity used by resume/checkpoint logic."""

    value = record.get("id")
    if value is None:
        raise ValueError("Every GRIP record must contain an id for resumable execution.")
    return str(value)


def _question_key(value) -> str:
    """Normalize the question field for exact graph-output recovery."""

    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value)


def _checkpoint_path(output_file: str | Path) -> Path:
    path = Path(output_file)
    return path.with_name(path.name + ".checkpoint.json")


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Rewrite a JSONL file durably, so a torn last line cannot poison resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl_durable(path: Path, rows: list[dict]) -> None:
    """Append one graph's predictions and force them to stable storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_checkpoint(path: Path, completed_ids: set[str], total_graphs: int) -> None:
    """Atomically persist graph-level progress after a complete graph."""

    payload = {
        "format_version": 1,
        "completed_graph_ids": sorted(completed_ids),
        "completed_graph_count": len(completed_ids),
        "total_graphs": total_graphs,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _recover_completed_graphs(
        input_data: list[dict],
        output_file: str,
        overwrite: bool,
) -> set[str]:
    """Recover only fully written graphs and discard partial/duplicate rows.

    The old implementation used ``len(output_lines)`` as the number of
    completed graphs.  That is unsafe because one graph contributes many
    question rows.  This routine treats a graph as complete only when the
    output contains exactly all expected questions for that graph.  Any
    partial or malformed tail is removed before the next run appends results.
    """

    output_path = Path(output_file)
    checkpoint_path = _checkpoint_path(output_file)
    if overwrite:
        for path in (output_path, checkpoint_path):
            if path.exists():
                path.unlink()
        return set()
    if not output_path.exists():
        return set()

    expected = {
        _graph_id(record): Counter(_question_key(question) for question in record.get("questions", []))
        for record in input_data
    }
    rows_by_graph: dict[str, list[dict]] = {}
    parsed_rows = 0
    malformed_rows = 0
    with output_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                graph_id = str(row["id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                malformed_rows += 1
                continue
            if graph_id not in expected:
                malformed_rows += 1
                continue
            rows_by_graph.setdefault(graph_id, []).append(row)
            parsed_rows += 1

    completed = set()
    for graph_id, rows in rows_by_graph.items():
        actual_questions = Counter(_question_key(row.get("question")) for row in rows)
        if actual_questions == expected[graph_id]:
            completed.add(graph_id)
    kept_rows = [
        row
        for record in input_data
        if (graph_id := _graph_id(record)) in completed
        for row in rows_by_graph[graph_id]
    ]
    if malformed_rows or len(kept_rows) != parsed_rows:
        _atomic_write_jsonl(output_path, kept_rows)

    _write_checkpoint(checkpoint_path, completed, len(input_data))
    print(
        f"Resume state: completed_graphs={len(completed)}/{len(input_data)}; "
        f"kept_prediction_rows={len(kept_rows)}",
        flush=True,
    )
    return completed


def extract_all_evidence_and_answers(text, add_evidence=True):
    if add_evidence:
        evidences = extract_tag_content(text, "evidence")
        evidences = "; ".join(evidences)
    else:
        evidences = ""

    answers = extract_tag_content(text, "answer")
    answers = "; ".join(answers)
    return evidences, answers


def run(
        rank: int,
        input_data: list[dict],
        exp_args: dict,
        training_args: TrainingArguments,
        output_file: str,
        ref_data: Optional[list[dict]] = None,
        completed_ids: Optional[set[str]] = None,
):
    run_started = time.time()
    os.environ.setdefault("RANK", str(rank))
    seed = training_args.seed + rank
    _ = set_random_seed(seed)
    completed_ids = set(completed_ids or set())
    pending_records = [
        (index, record)
        for index, record in enumerate(input_data)
        if _graph_id(record) not in completed_ids
    ]
    log_event("grip_run_start", rank=rank, seed=seed, input_records=len(input_data),
              completed_graphs=len(completed_ids), pending_graphs=len(pending_records),
              output_file=str(Path(output_file).resolve()),
              reference_records=len(ref_data) if ref_data is not None else 0,
              experiment_config=exp_args, training_config=training_args.to_dict())

    if not pending_records:
        log_event("grip_run_skipped_complete", completed=len(completed_ids), total=len(input_data))
        return

    # Generate all tasks before loading the fine-tuning model. This prevents the
    # task generator and the trainable base model from occupying GPU memory at once.
    task_started = time.time()
    log_event("grip_task_generation_start", pending_graphs=len(pending_records),
              model_name=exp_args.get("model_name"), task_generator_model_name=exp_args.get("task_generator_model_name"),
              task_cache_dir=str(Path(exp_args.get("task_cache_dir", "")).resolve()) if exp_args.get("task_cache_dir") else None)
    tokenizer = get_hf_ft_tokenizer(**exp_args)
    # Keep the original full graph order while generating tasks.  This makes
    # prompt construction and random subgraph sampling stable across resumes;
    # only the later train/infer loop skips completed graphs.
    graph_list = [data["graph"] for data in input_data]
    title_list = [data["title"] for data in input_data]
    task_data = GripTaskGeneration(graph_list, title_list, tokenizer=tokenizer, refer_data=ref_data, **exp_args)()
    log_event("grip_task_generation_complete", task_sets=len(task_data),
              context_samples=[len(item.context_samples) for item in task_data],
              qa_samples=[len(item.qa_samples) for item in task_data],
              elapsed_seconds=round(time.time() - task_started, 3))
    timer = Timer()
    # start training
    for (i, input_d) in tqdm(
            pending_records,
            total=len(pending_records),
            desc=f"Rank {rank}, Sample: "):
        task_d = task_data[i]
        graph_started = time.time()
        log_event("grip_graph_start", rank=rank, graph_index=i, graph_id=input_d.get("id"),
                  title=input_d.get("title"), context_samples=len(task_d.context_samples),
                  qa_samples=len(task_d.qa_samples))
        log_event("grip_finetune_model_loading", graph_index=i, model_name=exp_args.get("model_name"),
                  model_source=exp_args.get("model_source"), load_dir=exp_args.get("load_dir"))
        model, tokenizer = get_ft_model(**exp_args)
        log_event("grip_finetune_model_ready", graph_index=i, **model_summary(model))
        log_cuda_memory("grip_memory_before_training")
        training_started = time.time()
        log_event("grip_training_start", graph_index=i, training_args=training_args.to_dict(),
                  lora_r=exp_args.get("lora_r"), lora_alpha=exp_args.get("lora_alpha"),
                  target_modules=exp_args.get("target_modules"))
        model, tokenizer = train(
            model=model,
            tokenizer=tokenizer,
            training_dataset=task_d,
            training_args=training_args,
            eval_dataset=None,
            **exp_args,
        )
        log_event("grip_training_complete", graph_index=i,
                  elapsed_seconds=round(time.time() - training_started, 3), **model_summary(model))
        log_cuda_memory("grip_memory_after_training")

        # Inference immediately follows training for this graph, matching GRIP's
        # per-graph train-then-infer protocol.
        if exp_args["continue_training"]:
            model = model.merge_and_unload()
        if model.device.type == "cpu":
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        eval_dataset = gen_eval_task(
            eval_mode="grip",
            input_data=input_d,
            tokenizer=tokenizer,
            **exp_args,
        )
        output_results = []
        output_nosample_results = []
        log_event("grip_inference_start", graph_index=i, graph_id=input_d.get("id"),
                  question_count=len(eval_dataset), max_new_tokens=exp_args.get("gen_max_length"),
                  do_sample=False, model_device=str(model.device))
        inference_started = time.time()
        timer.start()
        for j in trange(0, len(eval_dataset), 1, desc=f"Inference.", disable=False, ):
            input_ids, Q, A = eval_dataset[j]
            if isinstance(A, str):
                A = [A]

            input_ids = input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            # output = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     pad_token_id=tokenizer.pad_token_id,
            #     eos_token_id=tokenizer.eos_token_id,
            #     max_new_tokens=exp_args["gen_max_length"],
            #     do_sample=exp_args["do_sample"],
            #     top_p=exp_args["top_p"],
            #     temperature=exp_args["temperature"],
            # )

            output_no_sample = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=exp_args["gen_max_length"],
                do_sample=False
            )
             
            output = output_no_sample


            response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            response_no_sample = tokenizer.decode(output_no_sample[0][input_ids.shape[-1]:], skip_special_tokens=True)
            evidence, answer = extract_all_evidence_and_answers(response, False)
            evidence, answer_no_sample = extract_all_evidence_and_answers(response_no_sample, False)

            output_result = {
                "id": input_d["id"],
                "question": Q,
                "raw_response": response.strip(),
                "response": answer.strip(),
                "raw_response_no_sample": response_no_sample.strip(),
                "response_no_sample": answer_no_sample.strip(),
                "evidence": evidence.strip(),
                "target": A}
            output_results.append(output_result)
        timer.end()
        log_event("grip_inference_complete", graph_index=i, predictions=len(output_results),
                  elapsed_seconds=round(time.time() - inference_started, 3))
        # Persist one prediction per JSONL line. This is consumable by both the
        # direct runner and mp_wrapper, while still allowing resume by graph ID.
        _append_jsonl_durable(Path(output_file), output_results)
        completed_ids.add(_graph_id(input_d))
        _write_checkpoint(_checkpoint_path(output_file), completed_ids, len(input_data))
        log_event("grip_output_saved", graph_index=i, output_file=str(Path(output_file).resolve()),
                  rows_written=len(output_results), completed_graphs=len(completed_ids),
                  graph_elapsed_seconds=round(time.time() - graph_started, 3))
        del model
        torch.cuda.empty_cache()
        log_cuda_memory("grip_memory_after_model_release")
    print(f"Total inference time: {timer.return_time()}")
    log_event("grip_run_complete", rank=rank, output_file=str(Path(output_file).resolve()),
              inference_seconds=round(timer.return_time(), 3), total_elapsed_seconds=round(time.time() - run_started, 3))


def main():
    log_runtime("grip_runtime_environment")
    base_args, training_args, exp_args = parse_args(
        (
            BaseArguments,
            TrainingArguments,
            (TaskArguments, ModelArguments, CustomTrainingArguments, InferenceArguments)),
        no_dict=(TrainingArguments,)
    )
    # set up experiment
    rank = base_args.pop("rank")
    input_file = base_args.pop("input_file")
    ref_file = base_args.pop("ref_file")
    output_file = base_args.pop("output_file")
    overwrite = base_args.pop("overwrite")
    os.environ["RANK"] = str(rank)
    log_event("grip_arguments_parsed", rank=rank, input_file=str(Path(input_file).resolve()),
              ref_file=str(Path(ref_file).resolve()) if ref_file else None,
              output_file=str(Path(output_file).resolve()), overwrite=overwrite,
              experiment_config=exp_args, training_config=training_args.to_dict())
    input_data = load_list_json(input_file)
    completed_ids = _recover_completed_graphs(input_data, output_file, overwrite)

    if os.path.exists(ref_file):
        ref_data = load_list_json(ref_file)
    else:
        ref_data = None
    run(rank, input_data, exp_args, training_args, output_file, ref_data, completed_ids)


if __name__ == "__main__":
    main()

"""Stage 1: generate and persist GRIP training tasks for one graph.

The expensive LLM-generated portions are checkpointed in --task-cache-dir.
The final task file is only published after every task category is complete.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from grip.tasks import GripTaskGeneration
from models import get_hf_ft_tokenizer
from utils import load_list_json, set_random_seed, log_event, log_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', default='outputs/data/nell23k/processed_test.json')
    parser.add_argument('--ref-file', default='outputs/data/nell23k/processed_val.json')
    parser.add_argument('--task-file', default='artifacts/nell23k_paper/tasks.json')
    parser.add_argument('--task-cache-dir', default='artifacts/task_cache/nell23k_paper_qwen')
    parser.add_argument('--overwrite-task-file', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model-name', default='qwen-7b')
    parser.add_argument('--model-source', choices=('local', 'modelscope', 'hf', 'auto'), default='local')
    parser.add_argument('--model-cache-dir', default='model_cache')
    parser.add_argument('--local-files-only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tokenize-max-length', type=int, default=4096)
    parser.add_argument('--task-generator-model-name', default='qwen-7b')
    parser.add_argument('--task-generator-use-vllm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--task-generator-batch-size', type=int, default=4)
    parser.add_argument('--task-gen-max-length', type=int, default=1000)
    parser.add_argument('--num-context-qa', type=int, default=8000)
    parser.add_argument('--num-reason-qa', type=int, default=2000)
    parser.add_argument('--num-summarization', type=int, default=6000)
    parser.add_argument('--sample-node-attribute-task', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--repharse-context-qa', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--context-upsampling', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--format-as-instruction', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--involve-qa-epochs', type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = __import__("time").time()
    log_runtime("stage1_runtime_environment")
    log_event("stage1_start", **vars(args))
    for name in ("input_file", "ref_file", "task_file", "task_cache_dir", "model_cache_dir"):
        value = Path(getattr(args, name)).expanduser().resolve()
        log_event("stage1_path", name=name, path=str(value), exists=value.exists())
    task_file = Path(args.task_file)
    if task_file.is_file() and not args.overwrite_task_file:
        with task_file.open('r', encoding='utf-8') as stream:
            existing = json.load(stream)
        log_event("stage1_task_file_reused", task_file=str(task_file.resolve()),
                  context_samples=len(existing['context_samples']), qa_samples=len(existing['qa_samples']),
                  elapsed_seconds=round(__import__("time").time() - started, 3))
        print(f'Task file already complete: {task_file.resolve()}')
        print(f"Context samples: {len(existing['context_samples'])}; QA samples: {len(existing['qa_samples'])}")
        return

    if not Path(args.input_file).is_file():
        raise FileNotFoundError(f'Input graph file does not exist: {args.input_file}')
    set_random_seed(args.seed)
    input_data = load_list_json(args.input_file)
    ref_data = load_list_json(args.ref_file) if Path(args.ref_file).is_file() else None
    log_event("stage1_input_loaded", input_file=str(Path(args.input_file).resolve()), graph_records=len(input_data),
              ref_file=str(Path(args.ref_file).resolve()), ref_records=len(ref_data) if ref_data is not None else 0)
    if len(input_data) != 1:
        raise ValueError(f'This staged NELL23K runner expects one graph record; found {len(input_data)}.')

    log_event("stage1_tokenizer_loading", model_name=args.model_name, model_source=args.model_source,
              model_cache_dir=str(Path(args.model_cache_dir).resolve()), local_files_only=args.local_files_only)
    tokenizer = get_hf_ft_tokenizer(
        model_name=args.model_name,
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        tokenize_max_length=args.tokenize_max_length,
    )
    graph = input_data[0]
    log_event("stage1_graph_loaded", graph_id=graph.get("id"), title=graph.get("title"),
              graph_nodes=len(graph["graph"].get("node_list", [])),
              graph_edges=len(graph["graph"].get("edge_list", [])))
    log_event("stage1_task_generation_start", task_generator_model_name=args.task_generator_model_name,
              task_generator_use_vllm=args.task_generator_use_vllm,
              task_generator_batch_size=args.task_generator_batch_size,
              task_cache_dir=str(Path(args.task_cache_dir).resolve()),
              num_context_qa=args.num_context_qa, num_reason_qa=args.num_reason_qa,
              num_summarization=args.num_summarization)
    generator = GripTaskGeneration(
        graph_list=[graph['graph']],
        title_list=[graph['title']],
        tokenizer=tokenizer,
        refer_data=ref_data,
        task_generator_model_name=args.task_generator_model_name,
        task_generator_use_vllm=args.task_generator_use_vllm,
        task_generator_batch_size=args.task_generator_batch_size,
        task_cache_dir=args.task_cache_dir,
        task_gen_max_length=args.task_gen_max_length,
        num_context_qa=args.num_context_qa,
        num_reason_qa=args.num_reason_qa,
        num_summarization=args.num_summarization,
        sample_node_attribute_task=args.sample_node_attribute_task,
        repharse_context_qa=args.repharse_context_qa,
        context_upsampling=args.context_upsampling,
        format_as_instruction=args.format_as_instruction,
        involve_qa_epochs=args.involve_qa_epochs,
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
    )
    dataset = generator()[0]
    log_event("stage1_task_generation_complete", context_samples=len(dataset.context_samples),
              qa_samples=len(dataset.qa_samples), elapsed_seconds=round(__import__("time").time() - started, 3))
    payload = {
        'format_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed,
        'source': {
            'input_file': str(Path(args.input_file).resolve()),
            'graph_id': graph['id'],
            'title': graph['title'],
            'graph_edges': len(graph['graph']['edge_list']),
            'graph_nodes': len(graph['graph']['node_list']),
        },
        'generation_config': {
            'num_context_qa': args.num_context_qa,
            'num_reason_qa': args.num_reason_qa,
            'num_summarization': args.num_summarization,
            'context_upsampling': args.context_upsampling,
            'format_as_instruction': args.format_as_instruction,
            'task_cache_dir': str(Path(args.task_cache_dir).resolve()),
        },
        'context_samples': dataset.context_samples,
        'qa_samples': dataset.qa_samples,
    }
    task_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = task_file.with_suffix(task_file.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False)
    os.replace(temporary, task_file)
    print('\n=== Stage 1 complete: training tasks persisted ===')
    print(f'Task file:       {task_file.resolve()}')
    print(f'Context samples: {len(dataset.context_samples)}')
    print(f'QA samples:      {len(dataset.qa_samples)}')
    log_event("stage1_complete", task_file=str(task_file.resolve()),
              context_samples=len(dataset.context_samples), qa_samples=len(dataset.qa_samples),
              elapsed_seconds=round(__import__("time").time() - started, 3))


if __name__ == '__main__':
    main()

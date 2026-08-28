"""Stage 3: load a saved GRIP LoRA adapter, infer, and evaluate NELL23K."""

import argparse
import os
import time
from pathlib import Path

import torch
from tqdm.auto import trange

from evaluation import auto_eval_batch
from grip.tasks import gen_eval_task
from models.utils import get_hf_llm_tokenizer
from constants import MODELSCOPE_DECODER_ONLY_LLMS, HF_DECODER_ONLY_LLMS, TORCH_DTYPE
from utils import extract_tag_content, load_list_json, save_entry_to_list_json, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', default='outputs/data/nell23k/processed_test.json')
    parser.add_argument('--adapter-dir', default='artifacts/nell23k_paper/lora_adapter')
    parser.add_argument('--output-file', default='outputs/grip_inf/nell23k/nell23k_qwen_paper_3090.jsonl')
    parser.add_argument('--resume', action='store_true', help='Resume from an existing JSONL prediction file.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model-name', default='qwen-7b')
    parser.add_argument('--model-source', choices=('local', 'modelscope', 'hf', 'auto'), default='local')
    parser.add_argument('--model-cache-dir', default='model_cache')
    parser.add_argument('--local-files-only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tokenize-max-length', type=int, default=4096)
    parser.add_argument('--gen-max-length', type=int, default=1000)
    return parser.parse_args()


def answer_from_text(text: str) -> str:
    answers = extract_tag_content(text, 'answer')
    return '; '.join(answers).strip()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    input_rows = load_list_json(args.input_file)
    if len(input_rows) != 1:
        raise ValueError(f'This staged NELL23K runner expects one graph record; found {len(input_rows)}.')
    adapter_dir, output_file = Path(args.adapter_dir), Path(args.output_file)
    if not (adapter_dir / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Missing trained LoRA adapter: {adapter_dir}. Run Stage 2 first.')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if output_file.exists():
        if args.resume:
            completed = len(load_list_json(output_file))
        else:
            output_file.unlink()

    model_id = (MODELSCOPE_DECODER_ONLY_LLMS[args.model_name]
                if args.model_source in {'local', 'modelscope', 'auto'}
                else HF_DECODER_ONLY_LLMS[args.model_name])
    model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        load_dir=str(adapter_dir),
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        tokenize_max_length=args.tokenize_max_length,
        dtype=TORCH_DTYPE['bfloat16'],
        peft=False,
        device_map='auto',
    )
    model.eval()
    eval_dataset = gen_eval_task(
        eval_mode='grip', input_data=input_rows[0], tokenizer=tokenizer,
        no_graph_context=True, use_subgraph=False, index_format=False,
    )
    if completed > len(eval_dataset):
        raise ValueError(f'Existing output has {completed} rows but evaluation only has {len(eval_dataset)} questions.')
    print('\n=== Stage 3: inference and evaluation ===')
    print(f'Resuming at {completed}/{len(eval_dataset)} predictions')
    started = time.time()
    for index in trange(completed, len(eval_dataset), desc='Inference', unit='question', dynamic_ncols=True):
        input_ids, question, target = eval_dataset[index]
        if isinstance(target, str):
            target = [target]
        input_ids = input_ids.to(model.device)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.gen_max_length,
            do_sample=False,
        )
        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        save_entry_to_list_json(output_file, {
            'id': input_rows[0]['id'],
            'question': question,
            'raw_response': response,
            'response': answer_from_text(response),
            'target': target,
        })

    rows = load_list_json(output_file)
    metrics = auto_eval_batch(
        preds=[row['response'] for row in rows],
        targets=[row['target'] for row in rows],
        metrics=['em', 'f1', 'hit'],
    )
    print('\n=== Stage 3 complete ===')
    print(f'Predictions: {output_file.resolve()}')
    print(f'Elapsed seconds: {time.time() - started:.2f}')
    for name, value in metrics.items():
        print(f'{name}: {float(value):.6f}')


if __name__ == '__main__':
    main()

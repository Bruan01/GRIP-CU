"""Stage 2: train and save one GRIP LoRA adapter from a persisted task file."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import TrainingArguments

from grip.tasks.train_tasks.task_dataset import TaskDataset
from grip.training import train
from models import get_ft_model
from utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-file', default='artifacts/nell23k_paper/tasks.json')
    parser.add_argument('--adapter-dir', default='artifacts/nell23k_paper/lora_adapter')
    parser.add_argument('--trainer-output-dir', default='artifacts/nell23k_paper/trainer')
    parser.add_argument('--overwrite-adapter', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model-name', default='qwen-7b')
    parser.add_argument('--model-source', choices=('local', 'modelscope', 'hf', 'auto'), default='local')
    parser.add_argument('--model-cache-dir', default='model_cache')
    parser.add_argument('--local-files-only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tokenize-max-length', type=int, default=4096)
    parser.add_argument('--lora-r', type=int, default=4)
    parser.add_argument('--lora-alpha', type=int, default=8)
    parser.add_argument('--target-modules', nargs='+', default=['down_proj', 'up_proj', 'gate_proj'])
    parser.add_argument('--num-train-epochs', type=float, default=1)
    parser.add_argument('--involve-qa-epochs', type=float, default=10)
    parser.add_argument('--s1-stop-loss-threshold', type=float, default=0.15)
    parser.add_argument('--s2-stop-loss-threshold', type=float, default=0.15)
    parser.add_argument('--s1-min-epoch', type=float, default=1)
    parser.add_argument('--s2-min-epoch', type=float, default=1)
    parser.add_argument('--per-device-train-batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.98)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_file, adapter_dir = Path(args.task_file), Path(args.adapter_dir)
    if not task_file.is_file():
        raise FileNotFoundError(f'Missing completed task file: {task_file}. Run Stage 1 first.')
    if adapter_dir.exists() and any(adapter_dir.iterdir()) and not args.overwrite_adapter:
        print(f'Adapter already exists: {adapter_dir.resolve()}')
        print('Use --overwrite-adapter to retrain it.')
        return

    set_random_seed(args.seed)
    with task_file.open('r', encoding='utf-8') as stream:
        task_payload = json.load(stream)
    context_samples = task_payload['context_samples']
    qa_samples = task_payload['qa_samples']
    if not context_samples:
        raise ValueError('Task file contains no context samples; cannot run Stage 1 training.')
    if args.involve_qa_epochs > 0 and not qa_samples:
        raise ValueError('Task file contains no QA samples but Stage 2 epochs are positive.')

    model, tokenizer = get_ft_model(
        model_name=args.model_name,
        model_source=args.model_source,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        tokenize_max_length=args.tokenize_max_length,
        dtype='bfloat16',
        quantization=False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
    dataset = TaskDataset(context_samples=context_samples, qa_samples=qa_samples, tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=args.trainer_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='no',
        report_to='none',
        bf16=True,
        tf32=False,
        seed=args.seed,
        remove_unused_columns=True,
        lr_scheduler_type='linear',
    )
    print('\n=== Stage 2: LoRA training ===')
    print(f'Context samples: {len(context_samples)}')
    print(f'QA samples:      {len(qa_samples)}')
    print(f'Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}')
    model, tokenizer = train(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        training_dataset=dataset,
        involve_qa_epochs=args.involve_qa_epochs,
        gather_batches=False,
        s1_stop_loss_threshold=args.s1_stop_loss_threshold,
        s2_stop_loss_threshold=args.s2_stop_loss_threshold,
        s1_min_epoch=args.s1_min_epoch,
        s2_min_epoch=args.s2_min_epoch,
    )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # 修复：如果 PEFT 将适配器保存到了子目录（如 'mylora'），则将其展平到根目录
    # 这样可以确保推理脚本能直接在 adapter_dir 下找到 adapter_config.json
    mylora_dir = adapter_dir / "mylora"
    if mylora_dir.is_dir():
        for item in mylora_dir.iterdir():
            item.rename(adapter_dir / item.name)
        mylora_dir.rmdir()

    metadata = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'task_file': str(task_file.resolve()),
        'context_samples': len(context_samples),
        'qa_samples': len(qa_samples),
        'model_name': args.model_name,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'target_modules': args.target_modules,
        'stage1_epochs': args.num_train_epochs,
        'stage2_epochs': args.involve_qa_epochs,
        'seed': args.seed,
    }
    with (adapter_dir / 'grip_run_metadata.json').open('w', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    del model
    torch.cuda.empty_cache()
    print('\n=== Stage 2 complete: LoRA adapter saved ===')
    print(f'Adapter directory: {adapter_dir.resolve()}')


if __name__ == '__main__':
    main()

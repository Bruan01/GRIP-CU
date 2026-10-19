"""Compare the pre-refactor candidate score with the shared scorer."""

import argparse
import json
import random
from pathlib import Path

import torch

from constants import ANSWER_TAG, HF_DECODER_ONLY_LLMS, SYSTEM_PROMPT, TORCH_DTYPE
from grip.training.candidate_scorer import score_candidate_records
from models.utils import get_hf_llm_tokenizer


def _token_list(token_ids) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu()
        if token_ids.ndim == 2:
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids.tolist()]
    return [int(token_id) for token_id in token_ids]


def _legacy_scores(model, tokenizer, records, max_length: int) -> list[torch.Tensor]:
    scores = []
    device = next(model.parameters()).device
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for record in records:
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": record.get("system_prompt", SYSTEM_PROMPT)},
                {"role": "user", "content": record["user_content"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = _token_list(tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"])
        target_rows = [
            torch.tensor(_token_list(tokenizer(
                f"<{ANSWER_TAG}>{candidate}</{ANSWER_TAG}>",
                add_special_tokens=False,
            )["input_ids"]), dtype=torch.long)
            for candidate in record["candidates"]
        ]
        max_target_length = max(target.numel() for target in target_rows)
        prompt_length = min(len(prompt_ids), max(1, max_length - max_target_length))
        input_ids = torch.full(
            (len(target_rows), prompt_length + max_target_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, target in enumerate(target_rows):
            input_ids[row, :prompt_length] = torch.tensor(prompt_ids[:prompt_length], device=device)
            input_ids[row, prompt_length:prompt_length + target.numel()] = target.to(device)
            attention_mask[row, :prompt_length + target.numel()] = 1
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=max_target_length + 1,
        ).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        scores.append(torch.stack([
            log_probs[row, :target.numel()]
            .gather(1, target.to(device)[:, None])
            .squeeze(1)
            .mean()
            for row, target in enumerate(target_rows)
        ]).float().cpu())
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-file', required=True)
    parser.add_argument('--adapter-dir', required=True)
    parser.add_argument('--model-cache-dir', required=True)
    parser.add_argument('--sample-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-length', type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size < 5 or args.sample_size > 10:
        raise ValueError('--sample-size must be between 5 and 10.')
    with Path(args.candidate_file).open('r', encoding='utf-8') as stream:
        payload = json.load(stream)
    relation_records = [record for record in payload['records'] if record is not None]
    if len(relation_records) < args.sample_size:
        raise ValueError('Candidate file contains fewer relation records than requested.')
    selected = random.Random(args.seed).sample(relation_records, args.sample_size)

    model, tokenizer = get_hf_llm_tokenizer(
        model_name=HF_DECODER_ONLY_LLMS['qwen-7b'],
        load_dir=args.adapter_dir,
        model_source='local',
        model_cache_dir=args.model_cache_dir,
        local_files_only=True,
        tokenize_max_length=args.max_length,
        dtype=TORCH_DTYPE['bfloat16'],
        peft=False,
        device_map='auto',
    )
    model.eval()
    with torch.inference_mode():
        old_scores = _legacy_scores(model, tokenizer, selected, args.max_length)
        new_scores = score_candidate_records(
            model,
            tokenizer,
            selected,
            max_length=args.max_length,
            score_clip=None,
        ).scores
    errors = []
    for record, old, new in zip(selected, old_scores, new_scores):
        error = (old - new.detach().cpu()).abs()
        errors.append(error)
        print({
            'task_index': record['task_index'],
            'candidate_count': len(record['candidates']),
            'max_abs_error': float(error.max()),
            'max_abs_score': float(torch.maximum(old.abs(), new.detach().cpu().abs()).max()),
        })
    maximum_error = max(float(error.max()) for error in errors)
    print({
        'sample_size': len(selected),
        'max_abs_error': maximum_error,
        'tolerance': 1e-3,
        'pass': maximum_error <= 1e-3,
        'dtype': 'bfloat16 model weights; float32 log-softmax accumulation',
    })
    if maximum_error > 1e-3:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Stage 3: load a saved GRIP LoRA adapter, infer, and evaluate NELL23K."""

import argparse
import time
from pathlib import Path

import torch
from tqdm.auto import trange

from evaluation import auto_eval_batch
from grip.tasks import gen_eval_task
from models.utils import get_hf_llm_tokenizer
from constants import (
    MODELSCOPE_DECODER_ONLY_LLMS,
    HF_DECODER_ONLY_LLMS,
    TORCH_DTYPE,
    SYSTEM_PROMPT,
    ANSWER_TAG,
)
from utils import extract_tag_content, load_list_json, save_entry_to_list_json, set_random_seed
from grip.relation_cecr import (
    build_relation_memory_from_triples,
    load_relation_memory,
    parse_question_entities,
    parse_relation_candidates,
)
from grip.training.candidate_scorer import score_candidate_prompt


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
    parser.add_argument(
        '--decoding-mode', choices=('generate', 'candidate_score'), default='generate',
        help='Free generation (original baseline) or teacher-forced scoring over listed candidates.',
    )
    parser.add_argument('--relation-memory-file', default=None,
                        help='Optional CECR relation-memory JSON built from training triples.')
    parser.add_argument('--train-triples-file', default=None,
                        help='Build CECR relation memory from this TSV training graph (never use test edges).')
    parser.add_argument('--score-batch-size', type=int, default=8,
                        help='Number of candidate completions scored in one forward pass.')
    parser.add_argument('--memory-prior-weight', type=float, default=0.0,
                        help='Optional weight for CECR type compatibility prior; 0 keeps pure LM scores.')
    parser.add_argument('--evidence-weight', type=float, default=0.0,
                        help='Weight for train-only exact/pair-composition graph evidence.')
    parser.add_argument('--counterfactual-weight', type=float, default=0.0,
                        help='Weight for candidate-vs-best-alternative evidence contrast.')
    parser.add_argument(
        '--relation-card-mode', choices=('none', 'candidate'), default='none',
        help='Optionally append a training-only relation card to each candidate prompt.',
    )
    return parser.parse_args()


def answer_from_text(text: str) -> str:
    answers = extract_tag_content(text, 'answer')
    return '; '.join(answers).strip()


def build_relation_card_prompt_ids(tokenizer, title: str, question: str, card: str) -> torch.Tensor:
    """Rebuild the GRIP evaluation prompt with a card before generation."""
    user_content = (
        f"Given the context graph titled {title}, please answer the following question: {question} "
        f"Response in the following format:<{ANSWER_TAG}>[answer]</{ANSWER_TAG}>\n\n"
        f"{card}"
    )
    message = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_content},
    ]
    sample = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return tokenizer(sample, return_tensors='pt')['input_ids']


def score_candidates(
    model,
    tokenizer,
    prompt_ids: torch.Tensor | list[torch.Tensor],
    candidates: list[str],
    batch_size: int,
) -> list[float]:
    """Compatibility wrapper around the shared candidate scorer."""
    if not candidates:
        return []
    if isinstance(prompt_ids, list):
        if len(prompt_ids) != len(candidates):
            raise ValueError('One prompt is required for each candidate.')
        scores = []
        for prompt, candidate in zip(prompt_ids, candidates):
            result = score_candidate_prompt(
                model,
                tokenizer,
                prompt,
                [candidate],
                score_clip=20.0,
                candidate_batch_size=1,
            )
            scores.append(float(result.scores[0][0].detach().cpu()))
        return scores
    shared_prompt = prompt_ids[0] if prompt_ids.ndim == 2 else prompt_ids
    result = score_candidate_prompt(
        model,
        tokenizer,
        shared_prompt,
        candidates,
        score_clip=20.0,
        candidate_batch_size=batch_size,
    )
    return [float(score.detach().cpu()) for score in result.scores[0]]


def combine_cecr_scores(
    lm_scores: list[float],
    prior_scores: list[float],
    evidence_scores: list[float],
    memory_prior_weight: float = 0.0,
    evidence_weight: float = 0.0,
    counterfactual_weight: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Fuse LM, type, graph evidence, and counterfactual contrast scores."""
    if not (len(lm_scores) == len(prior_scores) == len(evidence_scores)):
        raise ValueError('LM, prior, and evidence scores must have the same length.')
    counterfactual_scores = [
        evidence - max(
            (other for index, other in enumerate(evidence_scores) if index != candidate_index),
            default=0.0,
        )
        for candidate_index, evidence in enumerate(evidence_scores)
    ]
    final_scores = [
        lm + memory_prior_weight * prior
        + evidence_weight * evidence
        + counterfactual_weight * contrast
        for lm, prior, evidence, contrast in zip(
            lm_scores, prior_scores, evidence_scores, counterfactual_scores
        )
    ]
    return final_scores, counterfactual_scores


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
    relation_memory = None
    if args.relation_memory_file:
        relation_memory = load_relation_memory(args.relation_memory_file)
    elif args.train_triples_file:
        relation_memory = build_relation_memory_from_triples(args.train_triples_file)
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
        candidates = parse_relation_candidates(question)
        extra = {}
        if args.decoding_mode == 'candidate_score' and candidates:
            if args.relation_card_mode == 'candidate':
                if relation_memory is None:
                    raise ValueError(
                        '--relation-card-mode candidate requires --relation-memory-file or --train-triples-file.'
                    )
                card_prompts = [
                    build_relation_card_prompt_ids(
                        tokenizer, input_rows[0]['title'], question,
                        relation_memory.render_card(relation, candidates),
                    )
                    for relation in candidates
                ]
                scoring_prompt = card_prompts
            else:
                scoring_prompt = input_ids
            lm_scores = score_candidates(
                model, tokenizer, scoring_prompt, candidates, args.score_batch_size,
            )
            source, target_entity = parse_question_entities(question)
            prior_scores = [
                relation_memory.relation_prior(relation, source, target_entity)
                if relation_memory is not None else 0.0
                for relation in candidates
            ]
            evidence_scores = [
                relation_memory.evidence_score(relation, source, target_entity)
                if relation_memory is not None else 0.0
                for relation in candidates
            ]
            final_scores, counterfactual_scores = combine_cecr_scores(
                lm_scores,
                prior_scores,
                evidence_scores,
                memory_prior_weight=args.memory_prior_weight,
                evidence_weight=args.evidence_weight,
                counterfactual_weight=args.counterfactual_weight,
            )
            best_index = max(range(len(candidates)), key=final_scores.__getitem__)
            response = candidates[best_index]
            ranked = sorted(
                zip(candidates, lm_scores, prior_scores, evidence_scores, counterfactual_scores, final_scores),
                key=lambda item: item[5], reverse=True,
            )
            extra = {
                'candidate_scores': [
                    {
                        'candidate': c,
                        'lm_score': lm,
                        'memory_prior': prior,
                        'evidence_score': evidence,
                        'counterfactual_score': contrast,
                        'score': score,
                    }
                    for c, lm, prior, evidence, contrast, score in zip(
                        candidates, lm_scores, prior_scores, evidence_scores, counterfactual_scores, final_scores
                    )
                ],
                'top_candidates': [
                    {'candidate': item[0], 'score': item[5]} for item in ranked[:5]
                ],
                'score_margin': float(ranked[0][5] - ranked[1][5]) if len(ranked) > 1 else None,
                'candidate_valid': True,
                'source_entity': source,
                'target_entity': target_entity,
                'evidence_details': [
                    relation_memory.evidence_detail(relation, source, target_entity)
                    if relation_memory is not None else {'score': 0.0}
                    for relation in candidates
                ],
                'confusion_set': (
                    list(relation_memory.card(response).confusing_relations)
                    if relation_memory is not None and relation_memory.card(response) is not None
                    else []
                ),
            }
            raw_response = f'<answer>{response}</answer>'
        else:
            if args.decoding_mode == 'candidate_score' and not candidates:
                extra['candidate_valid'] = False
            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.gen_max_length,
                do_sample=False,
            )
            raw_response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
            response = answer_from_text(raw_response)
        save_entry_to_list_json(output_file, {
            'id': input_rows[0]['id'],
            'question': question,
            'raw_response': raw_response,
            'response': response,
            'target': target,
            'decoding_mode': args.decoding_mode,
            'relation_card_mode': args.relation_card_mode,
            **extra,
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
    if args.decoding_mode == 'candidate_score':
        valid = sum(bool(row.get('candidate_valid', False)) for row in rows)
        print(f'candidate_valid_rate: {valid / max(1, len(rows)):.6f}')


if __name__ == '__main__':
    main()

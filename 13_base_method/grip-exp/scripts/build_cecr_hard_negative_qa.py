"""Build candidate-aware CECR QA with relation-specific hard negatives.

The generated QA uses the same wording and answer tags as NELL23K evaluation,
but samples edges from ``train.txt`` only.  Hard negatives are drawn first
from the confusion groups observed in the original GRIP 7B failures and then
from relations with compatible source/target type prototypes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grip.relation_cecr import (  # noqa: E402
    build_relation_memory_from_triples,
    infer_entity_type,
)


Q_FORMAT = (
    "What is the relation between word node {src} and word node {tgt}? "
    "Selected from the following candidate answers: {candidates}."
)
TEMPLATE = (
    "Given the context graph titled {title}, please answer the following question: "
    "{question} Response in the following format:<answer>[answer]</answer>"
)
ANSWER_FORMAT = "<answer>{answer}</answer>"
SYSTEM_PROMPTS = [
    "You are a thorough assistant. Your role is to offer concise, reliable, and clear guidance in response to user prompts. Do not speculate.",
    "You are a resourceful assistant. Your task is to provide direct, concise, and accurate guidance to user questions. Do not make assumptions.",
    "You are an insightful assistant. Your role is to give clear, concise, and correct responses to user prompts. Do not guess.",
]


def read_train(path: str | Path) -> list[tuple[str, str, str]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                rows.append((fields[0], fields[1], fields[2]))
    if not rows:
        raise ValueError(f"No valid TSV triples found in {path}")
    return rows


def _compatible(memory, relation: str, source: str, target: str) -> bool:
    card = memory.card(relation)
    if card is None:
        return False
    source_type = infer_entity_type(source)
    target_type = infer_entity_type(target)
    return source_type in card.source_types and target_type in card.target_types


def hard_negative_pool(memory, target_relation: str, source: str, target: str, all_relations: list[str]) -> list[str]:
    card = memory.card(target_relation)
    ordered = []
    if card is not None:
        ordered.extend(item for item in card.confusing_relations if item in all_relations)
    compatible = [
        relation for relation in all_relations
        if relation != target_relation and relation not in ordered and _compatible(memory, relation, source, target)
    ]
    # Stable ordering makes the generated artifact reproducible before RNG
    # shuffling of the final candidate list.
    return ordered + sorted(compatible)


def build_candidates(rng, memory, target_relation, source, target, all_relations, way):
    pool = hard_negative_pool(memory, target_relation, source, target, all_relations)
    pool = [relation for relation in pool if relation != target_relation]
    if len(pool) < way - 1:
        remainder = [relation for relation in all_relations if relation != target_relation and relation not in pool]
        pool.extend(remainder)
    if not pool:
        raise ValueError("Cannot construct a negative candidate for a relation set with one relation.")
    if len(pool) >= way - 1:
        negatives = list(rng.choice(pool, size=way - 1, replace=False))
    else:
        negatives = list(pool) + list(rng.choice(pool, size=way - 1 - len(pool), replace=True))
    return list(rng.permutation(negatives + [target_relation]))


def render_sample(title, question, answer, rng, card=None):
    system = rng.choice(SYSTEM_PROMPTS)
    user_prompt = TEMPLATE.format(title=title, question=question)
    if card:
        user_prompt += "\n\n" + card
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{ANSWER_FORMAT.format(answer=answer)}<|im_end|>\n\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-triples-file', default='outputs/data/nell23k/train.txt')
    parser.add_argument('--base-task-file', default='', help='Optional task file whose context_samples are preserved.')
    parser.add_argument('--out-task-file', default='artifacts/nell23k_cecr_hard_negative/tasks.json')
    parser.add_argument('--num-qa', type=int, default=12014)
    parser.add_argument('--way', type=int, default=10)
    parser.add_argument('--title', default='nell23k')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--preview', type=int, default=3)
    parser.add_argument(
        '--relation-card-mode', choices=('none', 'gold'), default='none',
        help='Append the gold relation card during training; use with CECR card-mode inference.',
    )
    args = parser.parse_args()

    triples = read_train(args.train_triples_file)
    memory = build_relation_memory_from_triples(triples)
    all_relations = sorted(memory.cards)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(
        len(triples), size=args.num_qa, replace=len(triples) < args.num_qa,
    )

    qa_samples = []
    hard_negative_counts = []
    for index in sample_indices:
        source, relation, target = triples[int(index)]
        candidates = build_candidates(
            rng, memory, relation, source, target, all_relations, args.way,
        )
        question = Q_FORMAT.format(
            src=source, tgt=target, candidates='; '.join(candidates),
        )
        card = (memory.render_card(relation, candidates) if args.relation_card_mode == 'gold' else None)
        qa_samples.append(render_sample(args.title, question, relation, rng, card=card))
        card = memory.card(relation)
        hard_negative_counts.append(sum(item in (card.confusing_relations if card else ()) for item in candidates))

    context_samples = []
    if args.base_task_file and Path(args.base_task_file).is_file():
        with open(args.base_task_file, encoding='utf-8') as handle:
            context_samples = json.load(handle).get('context_samples', [])

    payload = {
        'format_version': 3,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed,
        'source': {
            'train_triples_file': str(Path(args.train_triples_file).resolve()),
            'title': args.title,
            'train_triples': len(triples),
            'relations': len(all_relations),
        },
        'generation_config': {
            'num_qa': len(qa_samples),
            'way': args.way,
            'negative_policy': 'confusion_set_then_type_compatible_then_random',
            'relation_card_mode': args.relation_card_mode,
            'mean_confusion_negatives': float(np.mean(hard_negative_counts)),
        },
        'context_samples': context_samples,
        'qa_samples': qa_samples,
    }
    output = Path(args.out_task_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(output)
    print(f'Wrote {len(qa_samples)} hard-negative QA samples to {output}')
    print(f'Relations: {len(all_relations)}; mean confusion negatives: {np.mean(hard_negative_counts):.3f}')
    for sample in qa_samples[:args.preview]:
        print(sample)
        print('-' * 80)


if __name__ == '__main__':
    main()

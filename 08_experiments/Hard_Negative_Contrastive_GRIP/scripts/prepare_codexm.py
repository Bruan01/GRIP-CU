#!/usr/bin/env python3
"""Prepare official-rule CoDEx-M 10-way questions without modifying codexm/."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from download_codexm import FILES, GRIP_REV, CODEX_REV, verify_file
from prepare_fb15k237 import atomic_write_json, official_entity_text, sample_candidates, sha256_file
from prepare_wn18rr import load_triples

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {'train': 185584, 'validation': 10310, 'test': 10311}
SPLITS = {'train': 'train.txt', 'validation': 'valid.txt', 'test': 'test.txt'}
TEMPLATE = ('What is the relation between word node {src} and word node {tgt}? '
            'Selected from the following candidate answers: {candidates}.')


def prepare(raw, *, seed=2026, train_questions=2000, smoke_questions=96, validate_source=True):
    source_hashes = {name: sha256_file(raw / name) for name in FILES}
    if validate_source:
        for name in FILES:
            verify_file(raw / name, name)
    triples = {split: load_triples(raw / name) for split, name in SPLITS.items()}
    if validate_source and {s: len(t) for s, t in triples.items()} != COUNTS:
        raise ValueError('CoDEx-M split counts do not match the complete benchmark')
    if not 0 < smoke_questions <= min(train_questions, *(len(triples[s]) for s in ('validation', 'test'))):
        raise ValueError('smoke count must fit each split')
    if not 0 < train_questions <= len(triples['train']):
        raise ValueError('invalid training reference count')
    metadata = json.loads((raw / 'entity2text.json').read_text(encoding='utf-8'))
    entities = {key: official_entity_text(metadata[key]) for key in sorted(metadata)}
    if any(any(c in text for c in '\t\n\r') for text in entities.values()):
        raise ValueError('entity text cannot be represented losslessly by the TSV parser')
    rel_data = json.loads((raw / 'codexm.json').read_text(encoding='utf-8'))
    train_relations = {rel for _, rel, _ in triples['train']}
    relations = {key: rel_data[key]['label'] for key in sorted(train_relations)}
    if (validate_source and len(relations) != 51) or len(set(relations.values())) != len(relations):
        raise ValueError('expected unique training relation labels')
    order = sorted(relations.values())
    if len(order) < 10:
        raise ValueError('10-way prediction requires at least ten training labels')
    # Complete pinned mappings cover every positive split. Refuse silent fallback.
    for split, rows in triples.items():
        for src, rel, tgt in rows:
            if src not in entities or tgt not in entities or rel not in relations:
                raise ValueError(f'{split}: missing entity or training relation mapping')
    ids = {key: i for i, key in enumerate(entities)}
    graph = {'node_list': list(entities.values()),
             'edge_index': [[ids[src], ids[tgt]] for src, _, tgt in triples['train']],
             'edge_list': [[entities[src], relations[rel], entities[tgt]] for src, rel, tgt in triples['train']]}

    def question(split, index, rng):
        src, rel, tgt = triples[split][index]
        candidates = sample_candidates(relations[rel], order, rng)
        return {'question_id': f'codexm:{split}:{index}', 'split': split,
                'question_type': 'CoDExMRelationPrediction', 'true_hop': None,
                'question': TEMPLATE.format(src=entities[src], tgt=entities[tgt],
                                            candidates='; '.join(candidates)),
                'answer': relations[rel], 'source_node': entities[src], 'target_node': entities[tgt],
                'source_entity_id': src, 'target_entity_id': tgt, 'source_relation_id': rel,
                'node_indices': [ids[src], ids[tgt]], 'source_file': SPLITS[split],
                'source_triple_index': index, 'candidate_relations': candidates,
                'candidate_source': 'official_grip_codexm_sorted_train_labels_gold_only_exclusion'}

    train_rng = np.random.RandomState(seed)
    indices = np.random.RandomState(seed).permutation(len(triples['train']))[:train_questions]
    selected = {'train': [question('train', int(i), train_rng) for i in indices]}
    evaluation_rng = np.random.RandomState(seed)
    for split in ('validation', 'test'):
        selected[split] = [question(split, i, evaluation_rng) for i in range(len(triples[split]))]
    for split, questions in selected.items():
        for q in questions:
            candidates = q['candidate_relations']
            if len(candidates) != 10 or len(set(candidates)) != 10 or candidates.count(q['answer']) != 1:
                raise ValueError('invalid 10-way candidate set')
            if set(candidates) - set(order):
                raise ValueError('candidate outside training vocabulary')
        if split != 'train' and [q['source_triple_index'] for q in questions] != list(range(len(triples[split]))):
            raise ValueError('evaluation dropped or reordered source triples')
    outputs = {}
    for split, name in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        outputs[f'processed_{name}.json'] = {'title': 'codexm', 'id': '0', 'graph': graph,
            'questions': [[q['question'], q['node_indices']] for q in selected[split]],
            'answers': [q['answer'] for q in selected[split]]}
    base = {'title': 'codexm', 'id': '0', 'dataset': 'CoDEx-M', 'graph': graph,
            'dataset_metadata': {'seed': seed, 'num_candidates': 10, 'graph_source': 'train.txt',
                'train_qa_source': 'train.txt', 'structural_distance_status': 'not computed; true_hop is null'}}
    outputs['recurrent_relation_prediction_full.json'] = {
        **base, 'recurrent_questions': [q for rows in selected.values() for q in rows]}
    outputs['recurrent_relation_prediction_smoke.json'] = {
        **base, 'recurrent_questions': [q for rows in selected.values() for q in rows[:smoke_questions]]}
    if any(sha256_file(raw / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError('raw source changed during preparation')
    manifest = {'format_version': 1, 'dataset': 'CoDEx-M', 'official_counts_validated': validate_source,
        'source_sha256': source_hashes, 'codex_revision': CODEX_REV, 'grip_revision': GRIP_REV,
        'official_processor': f'https://github.com/JiaruiFeng/GRIP/blob/{GRIP_REV}/data/raw_datasets/codexm/process.py',
        'seed': seed, 'numpy_version': np.__version__, 'num_candidates': 10,
        'relation_order': order, 'counts': {'source_triples': {s: len(t) for s, t in triples.items()},
            'graph_nodes': len(entities), 'graph_edges': len(triples['train']),
            'train_relation_texts': len(order), 'full_questions': {s: len(q) for s, q in selected.items()}},
        'candidate_rule': 'permutation(train labels excluding gold)[:9], then permutation(negatives + gold); do not exclude other train-known relations from EVALUATION candidates',
        'eval_rng_schedule': 'one RandomState(seed): complete validation, then complete test; no subsampling',
        'train_rng_schedule': 'independent RandomState(seed) for train-row selection and train candidates',
        'deviations': ['Sort train relation labels instead of list(set(...)); historical exact permutations may differ',
            'Sort entity IDs for stable node indices; official text values are unchanged',
            'Retain final source row without newline; reject blank or malformed rows',
            'Add independently sampled train-only references and smoke prefixes'],
        'verification': {'train_only_graph_and_references': True, 'full_validation_and_test': True,
            'source_ids_and_node_indices': True, 'ten_unique_training_labels_including_gold': True,
            'heldout_facts_not_used_for_training_negative_filter': True}}
    return outputs, manifest, entities, relations


def write_preparation(raw, out, **kwargs):
    outputs, manifest, entities, relations = prepare(raw, **kwargs)
    manifest['raw_outputs'] = {}
    manifest['outputs'] = {}
    manifest['raw_outputs']['rel2text.json'], _ = atomic_write_json(raw / 'rel2text.json', relations, jsonl=False)
    # TSV selects the existing full-text entity parser used by WN; no label-only aliases.
    text = ''.join(key + '\t' + value + '\n' for key, value in entities.items())
    dest = raw / 'entity2text.txt'
    if dest.exists() and dest.read_text(encoding='utf-8') != text:
        raise ValueError('existing derived TSV differs; use a new raw directory')
    if not dest.exists():
        staged = dest.with_suffix('.txt.tmp'); staged.write_text(text, encoding='utf-8'); staged.replace(dest)
    manifest['raw_outputs'][dest.name] = {'sha256': sha256_file(dest), 'bytes': dest.stat().st_size}
    for name, payload in outputs.items():
        manifest['outputs'][name], _ = atomic_write_json(out / name, payload, jsonl=True)
    atomic_write_json(out / 'preparation_manifest.json', manifest, jsonl=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_dir', type=Path, default=ROOT / 'data/raw_datasets/codexm')
    parser.add_argument('--out_dir', type=Path, default=ROOT / 'data/codexm')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    manifest = write_preparation(args.raw_dir, args.out_dir, seed=args.seed)
    print(json.dumps({'output_dir': str(args.out_dir), 'counts': manifest['counts'],
                      'official_counts_validated': manifest['official_counts_validated']}, indent=2))


if __name__ == '__main__':
    main()

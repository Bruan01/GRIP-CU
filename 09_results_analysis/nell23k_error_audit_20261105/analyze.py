"""CPU-only audit of frozen NELL predictions and existing sampler parameters.

Run with guardenv Python from any directory. No training, decoding, checkpoint
selection, source-bank mutation, or new test evaluation is performed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP = REPO / '08_experiments/Hard_Negative_Contrastive_GRIP'
RUNS = Path('/home/kieran/grip-cu-runs')
BASE = RUNS / 'stage2_three_dataset_20261030/nell23k/search'
REF = RUNS / 'nell23k_stage2_refinement_20261102'
ROUND2 = RUNS / 'nell23k_stage2_round2_20261104'
ASSETS = EXP / 'results/hpo/stage2_three_dataset_20261030/nell23k/assets'
DATA = REPO / '04_benchmark/data/nell23k'
sys.path.insert(0, str(EXP / 'src'))
from hard_negative_grip.counterfactuals import (  # noqa: E402
    load_train_relation_profiles, sample_structure_counterfactual_group,
)

spec = importlib.util.spec_from_file_location(
    'grip_normalizer', REPO / '13_base_method/grip-exp/evaluation/utils.py')
normalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalizer)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def candidates(row):
    text = row['question'].split('Selected from the following candidate answers: ', 1)[1]
    return [s.strip() for s in text.rstrip('.').split(';')]


def summary(group):
    n = len(group)
    correct = sum(r['correct'] for r in group)
    return dict(count=n, correct=correct, errors=n-correct,
                em=correct/n if n else None,
                outside_candidates=sum(not r['candidate_valid'] for r in group),
                outside_candidates_wrong=sum(not r['candidate_valid'] and not r['correct'] for r in group),
                outside_candidates_correct=sum(not r['candidate_valid'] and r['correct'] for r in group),
                wrong_in_candidates=sum(r['candidate_valid'] and not r['correct'] for r in group))


def paired(before, after):
    assert len(before) == len(after)
    for a, b in zip(before, after):
        assert (a['question_id'], a['question'], a['target']) == (
            b['question_id'], b['question'], b['target'])
    return dict(
        both_correct=sum(a['correct'] and b['correct'] for a, b in zip(before, after)),
        both_wrong=sum(not a['correct'] and not b['correct'] for a, b in zip(before, after)),
        rescue=sum(not a['correct'] and b['correct'] for a, b in zip(before, after)),
        regression=sum(a['correct'] and not b['correct'] for a, b in zip(before, after)),
        regression_outside_candidates=sum(a['correct'] and not b['correct'] and
            not b['candidate_valid'] for a, b in zip(before, after)),
    )


def main():
    paths = {
        'valid_b1': BASE / 'b1_full_validation/predictions.jsonl',
        'valid_step16': BASE / 'trials/trial_0001/step_016/full_validation/predictions.jsonl',
        'valid_step32': BASE / 'trials/trial_0001/step_032/full_validation/predictions.jsonl',
        'valid_refine64': REF / 'trials/trial_0000/step_032/full_validation/predictions.jsonl',
        'valid_lambda005': ROUND2 / 'trials/trial_0000/step_032/full_validation/predictions.jsonl',
        'valid_lr0006': ROUND2 / 'trials/trial_0001/step_032/full_validation/predictions.jsonl',
        'test_b1': BASE / 'b1_test/predictions.jsonl',
        'test_step32': BASE / 'selected_test/predictions.jsonl',
        'test_refine64': REF / 'selected_test/predictions.jsonl',
    }
    predictions = {name: list(rows(path)) for name, path in paths.items()}
    sources = {split: read(DATA / f'processed/processed_{split}.json') for split in ('val', 'test')}
    validations = {}
    cache_paths = {}
    for name, group in predictions.items():
        source = sources['val' if name.startswith('valid') else 'test']
        assert len(group) == len(source['questions']) == len({r['question_id'] for r in group})
        for i, r in enumerate(group):
            assert r['question_id'] == f"0:test:{i}"
            assert r['question'] == source['questions'][i][0]
            target = source['answers'][i]
            assert r['target'] == (target if isinstance(target, list) else [target])
            norm = normalizer.normalize_answer
            assert r['correct'] == any(norm(r['response']) == norm(t) for t in r['target'])
            assert r['candidate_valid'] == (r['response'] in candidates(r))
            assert all(t in candidates(r) for t in r['target'])
        assert read(paths[name].parent / 'summary.json')['all'] == {
            'count': len(group), 'em': summary(group)['em']}
        cache_path = paths[name].parent / 'cache_identity.json'
        cache = read(cache_path)
        source_path = DATA / f"processed/processed_{'val' if name.startswith('valid') else 'test'}.json"
        assert cache['eval_sha256'] == sha(source_path)
        assert cache['gen_max_length'] == 32 and cache['dtype'] == 'bfloat16'
        cache_paths[f'{name}_decoder_identity'] = cache_path
        validations[name] = 'unique IDs; source text/targets; gold in candidates; EM; candidate flags; saved summary; source hash and decoder identity verified'

    bank_path = ASSETS / 'k9/counterfactual_bank.jsonl'
    bank = list(rows(bank_path))
    bank_index = {r['question_id']: r for r in bank}
    qa_freq = Counter(r['gold_relation'] for r in bank)
    triples = [line.split('\t') for line in (DATA / 'raw/train.txt').read_text().splitlines()]
    graph_freq = Counter(r for h, r, t in triples)
    known_pairs = defaultdict(set)
    for h, r, t in triples:
        known_pairs[h, t].add(r)

    result = {
        'question': 'Can unchanged SC-CL improve NELL23K through existing hyperparameters?',
        'claim_type': 'computed',
        'parent': str(BASE / 'selection.json'),
        'execution': {'python': sys.executable, 'python_version': sys.version.split()[0],
            'device': 'CPU; existing predictions and frozen B1 scores only',
            'training_launched': False, 'new_inference_launched': False,
            'source_bank_modified': False},
        'comparability': 'Existing local source questions, targets, decoder and seed2026; validation and test separated despite reused 0:test IDs. Earlier test exposure remains exploratory.',
        'validation': validations,
        'summaries': {name: summary(group) for name, group in predictions.items()},
        'pairs': {}, 'slices': {},
    }
    comparisons = [('valid_b1', 'valid_step32'), ('test_b1', 'test_step32'),
        ('valid_step16', 'valid_step32'), ('valid_step32', 'valid_refine64'),
        ('test_step32', 'test_refine64'), ('valid_step32', 'valid_lambda005'),
        ('valid_step32', 'valid_lr0006')]
    for a, b in comparisons:
        result['pairs'][f'{a} -> {b}'] = paired(predictions[a], predictions[b])

    error_rows = []
    generic = {'concept:proxyfor', 'concept:mutualproxyfor', 'concept:subpartof'}
    for name in ('valid_step32', 'test_step32'):
        group = predictions[name]
        errors = [r for r in group if not r['correct']]
        invalid = [r for r in errors if not r['candidate_valid']]
        relation_stats = []
        for rel in sorted({r['target'][0] for r in group}):
            subset = [r for r in group if r['target'][0] == rel]
            relation_stats.append(dict(relation=rel, graph_train_count=graph_freq[rel],
                eligible_contrastive_qa_count=qa_freq[rel], **summary(subset)))
        relation_stats.sort(key=lambda x: (-x['errors'], x['relation']))
        confusions = []
        for (gold, pred), n in Counter((r['target'][0], r['response']) for r in errors).most_common():
            training = [r for r in bank if r['gold_relation'] == gold]
            confusions.append(dict(gold=gold, prediction=pred, errors=n,
                prediction_in_train_vocab=pred in graph_freq,
                training_qa_count=len(training),
                training_qa_with_this_negative=sum(pred in r['negative_relations'] for r in training)))
        frequency_slices = {}
        for label, low, high in [('0',0,0), ('1-5',1,5), ('6-10',6,10), ('11-20',11,20), ('21+',21,10**9)]:
            frequency_slices[label] = summary([r for r in group if low <= qa_freq[r['target'][0]] <= high])
        result['slices'][name] = dict(
            generic_relation_summary=summary([r for r in group if r['target'][0] in generic]),
            other_relation_summary=summary([r for r in group if r['target'][0] not in generic]),
            relation_stats=relation_stats, confusions=confusions,
            outside_candidates=dict(count=len(invalid),
                in_train_vocab=sum(r['response'] in graph_freq for r in invalid),
                outside_train_vocab=sum(r['response'] not in graph_freq for r in invalid),
                malformed_relation_strings=sum(not re.fullmatch(r'concept:[a-z0-9_]+', r['response']) for r in invalid),
                answer_tags_missing=sum(not re.fullmatch(r'<answer>.*</answer>', r['raw_response'], re.S) for r in invalid),
                most_common=Counter(r['response'] for r in invalid).most_common(20)),
            eligible_training_qa_frequency_slices=frequency_slices,
        )
        for r in errors:
            error_rows.append(dict(cohort=name, question_id=r['question_id'],
                gold=r['target'][0], prediction=r['response'],
                outside_candidates=not r['candidate_valid'],
                outside_train_vocab=r['response'] not in graph_freq,
                eligible_training_qa_count=qa_freq[r['target'][0]], question=r['question']))

    method_names = ['valid_step32', 'valid_refine64', 'valid_lambda005', 'valid_lr0006']
    stable = [predictions[method_names[0]][i] for i in range(4950)
        if all(not predictions[name][i]['correct'] for name in method_names)]
    result['stable_validation_errors'] = dict(count=len(stable),
        top_relations=Counter(r['target'][0] for r in stable).most_common(15),
        interpretation='Common errors across four existing configurations; no achievable oracle/ensemble score claimed.')

    # Change one existing sampling parameter at a time. Reuse exact production
    # implementation and train-only known-pair filtering, no GPU or new scorer.
    profiles = load_train_relation_profiles(DATA / 'raw')
    proposals = {'current': {}, 'violation_weight2': {'violation_weight': 2.0},
        'violation_weight4': {'violation_weight': 4.0},
        'uniform_mix_1over6': {'uniform_mix': 1/6},
        'sampling_temperature015': {'temperature': .15}}
    aggregates = {name: Counter() for name in proposals}
    score_path = ASSETS / 'b1_train_scores/candidate_scores.jsonl'
    for qid, stream in itertools.groupby(rows(score_path), key=lambda r: r['qa_id']):
        items = list(stream)
        for name, params in proposals.items():
            gold_row = next(r for r in items if r['is_gold'])
            known = known_pairs[gold_row['head_entity'], gold_row['tail_entity']]
            sampled, _ = sample_structure_counterfactual_group(items, profiles=profiles,
                known_relations=known, **params)
            if name == 'current':
                assert sampled['negative_relations'] == bank_index[qid]['negative_relations']
            a = aggregates[name]
            a['qa_count'] += 1
            for item in sampled['selected_negatives']:
                a['negative_count'] += 1
                a['rank_top10'] += item['candidate_rank_all'] <= 10
                a['rank_over50'] += item['candidate_rank_all'] > 50
                a['score_gap_ge2'] += item['score_gap'] >= 2
                a['score_gap_le0'] += item['score_gap'] <= 0
                a['sum_gap'] += item['score_gap']
                a['sum_violation'] += item['score_violation']
        if aggregates['current']['qa_count'] % 750 == 0:
            print('audited score groups', aggregates['current']['qa_count'], flush=True)
    assert aggregates['current']['qa_count'] == len(bank) == 2974
    result['offline_sampler_sensitivity'] = {}
    for name, a in aggregates.items():
        n = a['negative_count']
        result['offline_sampler_sensitivity'][name] = dict(parameters=proposals[name],
            **dict(a), top10_fraction=a['rank_top10']/n,
            over50_fraction=a['rank_over50']/n, mean_gap=a['sum_gap']/n,
            mean_violation=a['sum_violation']/n)
    result['offline_sampler_scope'] = (
        'Exact current-bank reproduction on all 2974 rows. Alternatives change one sampling parameter only, '
        'fixed K=9, seed2026, production sampler, train vocabulary, train-known-pair filters and frozen B1 scores. '
        'Hardness is a training-side proxy, not measured validation/test EM or evidence of guaranteed gain. '
        'Existing pairwise_confusion takes precedence over temperature in score_violation; sampling temperature '
        'still changes the softmax. No held-out errors are injected into the bank.')
    inputs = {**paths, **cache_paths, 'bank': bank_path, 'frozen_scores': score_path,
        'processed_val': DATA/'processed/processed_val.json',
        'processed_test': DATA/'processed/processed_test.json',
        'raw_train': DATA/'raw/train.txt',
        'sampler_source': EXP/'src/hard_negative_grip/counterfactuals.py',
        'analyzer': Path(__file__)}
    result['inputs'] = {name: dict(path=str(p), sha256=sha(p)) for name, p in inputs.items()}
    result['next_route'] = 'Analysis complete. Consider a separately budgeted validation-only sampler-weight experiment; current search stays closed and frozen test result stays unchanged.'
    (OUT/'evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (OUT/'errors.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(error_rows[0]))
        writer.writeheader()
        writer.writerows(error_rows)
    print(json.dumps({k: result[k] for k in ['summaries', 'pairs', 'stable_validation_errors',
        'offline_sampler_sensitivity']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

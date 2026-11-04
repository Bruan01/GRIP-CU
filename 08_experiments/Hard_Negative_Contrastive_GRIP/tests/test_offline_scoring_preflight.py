"""Unresolvable generated identities must not bypass true-relation filtering."""
import argparse
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from score_relation_candidates_offline import load_matchable
from hard_negative_grip.task_file import train_relation_alias_index
from hard_negative_grip.wn_pairs import wn_question_entity_pairs


def qa(question, answer='hypernym'):
    return ('<|im_start|>user\n' + question +
        '? Response in the following format:<answer>[answer]</answer><|im_end|>\n'
        '<|im_start|>assistant\n<answer>' + answer + '</answer><|im_end|>')


def test_identity_exclusion_preserves_full_task_and_original_ids(tmp_path):
    texts = [qa('What is the relation between word node salt and pepper and word node bird'),
             qa('What is the relation between the word nodes "salt" and "bird"'),
             qa('Is bird alive', 'yes'),
             qa('What is the relation between node bird and node animal')]
    task = tmp_path / 'tasks.json'
    task.write_text(json.dumps({'qa_samples': texts, 'context_samples': []}))
    before = task.read_bytes()
    entities = {'salt and pepper', 'bird', 'animal'}
    def pair(text):
        pairs = wn_question_entity_pairs(text, entities)
        return pairs[0] if len(pairs) == 1 else None
    excluded = []
    items = load_matchable(argparse.Namespace(task_file=task),
        train_relation_alias_index(['hypernym']), pair,
        require_entity_pair=True, excluded_items=excluded)
    assert [item['qa_id'] for item in items] == ['task_qa:0', 'task_qa:3']
    assert items[0]['entity_pair'] == ('salt and pepper', 'bird')
    assert excluded == [{'qa_id': 'task_qa:1', 'gold_relation': 'hypernym',
                         'reason': 'missing_or_ambiguous_full_entity_identity'}]
    assert task.read_bytes() == before


def test_ambiguous_identity_is_excluded_before_model_scoring(tmp_path):
    task = tmp_path / 'tasks.json'
    task.write_text(json.dumps({'qa_samples': [qa('relation between a and b and c')]}))
    entities = {'a', 'b and c', 'a and b', 'c'}
    def pair(text):
        pairs = wn_question_entity_pairs(text, entities)
        return pairs[0] if len(pairs) == 1 else None
    excluded = []
    assert len(wn_question_entity_pairs(qa('relation between a and b and c'), entities)) == 2
    assert load_matchable(argparse.Namespace(task_file=task),
        train_relation_alias_index(['hypernym']), pair,
        require_entity_pair=True, excluded_items=excluded) == []
    assert excluded[0]['qa_id'] == 'task_qa:0'


def test_exclusion_reaches_full_task_training_with_zero_weight(tmp_path):
    import search_validation_hparams as hpo
    from hard_negative_grip.task_file import build_qa_assets_from_task_texts, load_score_hard_manifest
    from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
    texts = [qa('relation between node bird and node animal'),
             qa('relation between the word nodes "bird" and "animal"', 'hypernym.')]
    task = tmp_path / 'tasks.json'
    task.write_text(json.dumps({'qa_samples': texts, 'context_samples': [],
                                'original_question_ids': ['task_qa:0', 'task_qa:1']}))
    score = tmp_path / 'confusion'; score.mkdir()
    metadata = score / 'metadata.json'
    metadata.write_text(json.dumps({'entity_pair_required': True, 'excluded_identity_count': 1,
                                    'dataset': str(task)}))
    audit = {'require_entity_pair': True, 'task_file_sha256': hpo.sha256(task),
             'eligible_before_sharding': 1, 'excluded_count': 1,
             'excluded': [{'qa_id': 'task_qa:1', 'gold_relation': 'hypernym',
                          'reason': 'missing_or_ambiguous_full_entity_identity'}]}
    (score / 'qa_eligibility.json').write_text(json.dumps(audit))
    sampler = tmp_path / 'sampler'; sampler.mkdir()
    source = sampler / 'calibrated.jsonl'
    source.write_text(json.dumps({'question_id': 'task_qa:0', 'positive_relation': 'hypernym',
        'negative_relations': ['other'], 'hard_negative_relations': ['other'], 'uniform_negative_relations': [],
        'lambda_candidate': .5, 'shared_pool': 'train_only_valid_negatives',
        'negative_source': 'shared_valid_pool_calibrated', 'variant': 'calibrated'}) + '\n')
    (sampler / 'policy.json').write_text(json.dumps({'source_metadata': str(metadata), 'filter_splits': ['train']}))
    raw = tmp_path / 'raw'; raw.mkdir()
    (raw / 'train.txt').write_text('1 _hypernym 2\n1 _other 3\n')
    relation_file = raw / 'rel2text.json'
    relation_file.write_text(json.dumps({'_hypernym': 'hypernym', '_other': 'other'}))
    for method in ('soft_mix', 'calibrated'):
        output = tmp_path / (method + '.jsonl')
        hpo.freeze_manifest(source, output, {'method': method, 'lambda_scale': .5})
        manifest = load_score_hard_manifest(output)
        kept, metas = build_qa_assets_from_task_texts(texts, seed=2026,
            listed_negative_source='score_hard', relation_order=['hypernym', 'other'],
            score_hard_manifest=manifest)
        assert kept == texts
        assert metas[0]['listed_relations'] == ['other']
        assert metas[1]['listed_relations'] == [] and metas[1]['lambda_candidate'] == 0
        wired = verify_listed_manifest_wiring(task_path=task, manifest_paths={method: output},
            raw_dir=raw, expected_n=2, expected_listed=1, variants=(method,), relation_text_file=relation_file)
        assert wired['variants'][method]['n_generation_only_excluded'] == 1
    corrupted = json.loads(output.read_text().splitlines()[-1])
    corrupted['lambda_candidate'] = .1
    output.write_text(output.read_text().splitlines()[0] + '\n' + json.dumps(corrupted) + '\n')
    with pytest.raises(ValueError, match='generation-only'):
        verify_listed_manifest_wiring(task_path=task, manifest_paths={method: output},
            raw_dir=raw, expected_listed=1, variants=(method,), relation_text_file=relation_file)
    task.write_text(task.read_text() + ' ')
    with pytest.raises(ValueError, match='audit'):
        hpo.freeze_manifest(source, tmp_path / 'changed-task.jsonl', {'method': 'calibrated', 'lambda_scale': 1})

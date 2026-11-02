"""CPU checks for WN full-split evaluation and text-identity filtering."""
import json
from pathlib import Path
import random
import sys
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import run_wn_three_configs as wn
from hard_negative_grip.wn_pairs import load_wn_entity_text, wn_question_entity_pairs
from hard_negative_grip.fb_pairs import fb_known_pair_relations
from hard_negative_grip.shared_pool_samplers import sample_soft_mix


def test_wn_validation_counts_and_no_test_selection(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps({'validation': {'count': 3034, 'official_em': .6}}))
    assert wn.full_validation(path) == .6
    for payload in ({'validation': {'count': 10000, 'official_em': .6}},
                    {'validation': {'count': 3034, 'official_em': .6}, 'test': {}}):
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            wn.full_validation(path)


def test_wn_text_collisions_filter_train_only(tmp_path):
    (tmp_path / 'entity2text.txt').write_text('1\tsalt and pepper\n2\tword sense\n3\tword sense\n')
    (tmp_path / 'train.txt').write_text('1 _hypernym 2\n1 _also_see 3\n')
    entities = load_wn_entity_text(tmp_path)
    prompt = 'What is the relation between word node salt and pepper and word node word sense? Response in the following format'
    from hard_negative_grip.offline_scoring import parse_matchable_relation_qa
    from hard_negative_grip.task_file import train_relation_alias_index
    qa = '<|im_start|>user\n' + prompt + '\n<|im_end|>\n<|im_start|>assistant\n<answer>also see</answer><|im_end|>'
    parsed = parse_matchable_relation_qa(0, qa, train_relation_alias_index(['also see', 'hypernym']),
        pair_fn=lambda text: wn_question_entity_pairs(text, set(entities.values()))[0])
    assert parsed['matched_train_relation'] == 'also see'
    assert parsed['entity_pair'] == ('salt and pepper', 'word sense')
    pairs = wn_question_entity_pairs(prompt, set(entities.values()))
    assert pairs == [('salt and pepper', 'word sense')]
    known = fb_known_pair_relations(tmp_path, entity_text=entities,
        relation_text={'_hypernym': 'hypernym', '_also_see': 'also see'}, splits=('train',))
    assert known[pairs[0]] == {'hypernym', 'also see'}
    assert wn_question_entity_pairs('relation between unknown and word sense? Response in the following format', set(entities.values())) == []


def test_small_filtered_pools_are_unique_and_capped():
    for size in (1, 4, 8, 10):
        pool = [{'candidate_relation': str(i), 'candidate_score': -i,
                 'negative_mass': 1 / size} for i in range(size)]
        uniform, soft = sample_soft_mix(pool, k_uniform=6, k_soft=3, rho=.33,
                                        rng=random.Random(2026), ranked=pool, soft_pool_k=9)
        names = [r['candidate_relation'] for r in uniform + soft]
        assert len(names) == min(9, size)
        assert len(set(names)) == len(names)


def test_wn_steps_are_train_only_and_resumable(tmp_path):
    stages = {n: c for n, c, _ in wn.steps(tmp_path, tmp_path / 'wnraw', tmp_path / 'wndata')}
    assert stages['tasks'][stages['tasks'].index('--ref-file') + 1].endswith('processed_train.json')
    assert stages['score'][stages['score'].index('--filter_splits') + 1] == 'train'
    assert '--resume' in stages['score']
    assert stages['b1'][stages['b1'].index('--save_steps') + 1] == '10'
    assert stages['b1'][stages['b1'].index('--involve_qa_epochs') + 1] == '3'
    assert '--official_drop_last_line' not in stages['b1']

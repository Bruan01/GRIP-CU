"""Unresolvable generated identities must not bypass true-relation filtering."""
import argparse
import json
from pathlib import Path
import sys

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

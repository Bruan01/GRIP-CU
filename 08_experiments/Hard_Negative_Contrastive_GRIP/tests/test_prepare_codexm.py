"""Official candidate RNG, train-only provenance, and offline source integrity."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import download_codexm as download
import prepare_codexm as prep
from hard_negative_grip.fb_pairs import fb_known_pair_relations
from hard_negative_grip.wn_pairs import load_wn_entity_text, wn_question_entity_pairs


def fixture(raw):
    raw.mkdir()
    entities = {'e3': {'label': 'animal', 'description': 'a living thing'},
                'e1': {'label': 'salt and pepper', 'description': 'two things'},
                'e2': {'label': 'bird', 'description': 'a winged animal'}}
    (raw / 'entity2text.json').write_text(json.dumps(entities))
    (raw / 'codexm.json').write_text(json.dumps({f'r{i}': {'label': f'relation {i}'} for i in range(10)}))
    # Two train-known positive labels for the same text pair. Evaluation must
    # still include the other label; TRAIN negative filtering must exclude it.
    train = [('e1', f'r{i}', 'e2') for i in range(10)] + [('e2', 'r0', 'e3')]
    splits = {'train': train, 'validation': [('e1', 'r0', 'e2'), ('e3', 'r1', 'e1')],
              'test': [('e2', 'r2', 'e1'), ('e1', 'r3', 'e3')]}
    for split, name in prep.SPLITS.items():
        (raw / name).write_text('\n'.join('\t'.join(t) for t in splits[split]))
    return splits


def test_official_rng_and_train_only_graph(tmp_path):
    raw = tmp_path / 'raw'; triples = fixture(raw)
    outputs, manifest, entities, relations = prep.prepare(raw, train_questions=3,
        smoke_questions=1, validate_source=False)
    full = outputs['recurrent_relation_prediction_full.json']
    assert full['graph']['edge_list'] == [[entities[a], relations[r], entities[b]] for a, r, b in triples['train']]
    assert len(full['graph']['edge_list']) == 11  # preserve parallel relations
    assert manifest['official_counts_validated'] is False
    rng = np.random.RandomState(2026)
    order = sorted(relations.values())
    evaluation = [q for q in full['recurrent_questions'] if q['split'] != 'train']
    for q in evaluation:
        pool = [r for r in order if r != q['answer']]
        expected = rng.permutation(rng.permutation(pool)[:9].tolist() + [q['answer']]).tolist()
        assert q['candidate_relations'] == expected
        src, rel, tgt = triples[q['split']][q['source_triple_index']]
        assert (q['source_entity_id'], q['source_relation_id'], q['target_entity_id']) == (src, rel, tgt)
        assert q['node_indices'] == [list(entities).index(src), list(entities).index(tgt)]
    assert 'relation 1' in evaluation[0]['candidate_relations']
    changed, _, _, _ = prep.prepare(raw, train_questions=4, smoke_questions=1, validate_source=False)
    assert evaluation == [q for q in changed['recurrent_relation_prediction_full.json']['recurrent_questions'] if q['split'] != 'train']
    assert [q['question_id'] for q in evaluation] == ['codexm:validation:0', 'codexm:validation:1', 'codexm:test:0', 'codexm:test:1']


def test_tsv_full_text_parser_and_train_filter(tmp_path):
    raw = tmp_path / 'raw'; fixture(raw); out = tmp_path / 'data'
    first = prep.write_preparation(raw, out, train_questions=3, smoke_questions=1, validate_source=False)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert first == prep.write_preparation(raw, out, train_questions=3, smoke_questions=1, validate_source=False)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    entities = load_wn_entity_text(raw)
    full = json.loads((out / 'recurrent_relation_prediction_full.json').read_text())
    q = next(q for q in full['recurrent_questions'] if q['split'] == 'validation')
    prompt = prep.TEMPLATE.split('?')[0].format(src=q['source_node'], tgt=q['target_node'])
    pairs = wn_question_entity_pairs(prompt + '? Response in the following format', set(entities.values()))
    assert pairs == [(q['source_node'], q['target_node'])]
    known = fb_known_pair_relations(raw, entity_text=entities,
        relation_text=json.loads((raw / 'rel2text.json').read_text()), splits=('train',))
    assert known[pairs[0]] == {f'relation {i}' for i in range(10)}
    heldout_only = (entities['e3'], entities['e1'])
    assert heldout_only not in known


def test_production_refuses_fixture_and_pinned_source_tampering(tmp_path):
    raw = tmp_path / 'raw'; fixture(raw)
    with pytest.raises(ValueError, match='checksum'):
        prep.prepare(raw, train_questions=3, smoke_questions=1)
    original = (raw / 'train.txt').read_bytes()
    with pytest.raises(ValueError, match='checksum'):
        download.acquire(raw, offline=True)
    assert (raw / 'train.txt').read_bytes() == original
    assert download.canonical_hash({'x': 1, 'y': 2}) == download.canonical_hash({'y': 2, 'x': 1})
    assert download.canonical_hash({'label': 'bird'}) != download.canonical_hash({'label': 'birds'})


def test_malformed_rows_and_lossy_mapping_are_rejected(tmp_path):
    raw = tmp_path / 'raw'; fixture(raw)
    with (raw / 'train.txt').open('a') as handle:
        handle.write('\n\n')
    with pytest.raises(ValueError, match='blank'):
        prep.prepare(raw, train_questions=3, smoke_questions=1, validate_source=False)
    fixture_raw = tmp_path / 'raw2'; fixture(fixture_raw)
    values = json.loads((fixture_raw / 'entity2text.json').read_text())
    values['e1']['description'] = 'two\nlines'
    (fixture_raw / 'entity2text.json').write_text(json.dumps(values))
    with pytest.raises(ValueError, match='losslessly'):
        prep.prepare(fixture_raw, train_questions=3, smoke_questions=1, validate_source=False)

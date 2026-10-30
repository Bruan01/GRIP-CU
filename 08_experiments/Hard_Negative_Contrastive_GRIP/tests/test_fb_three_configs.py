"""CPU protocol checks for the fixed FB transfer experiment."""
from pathlib import Path
import sys
import json
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import run_fb_three_configs as fb


def test_training_arms_share_budget_and_keep_test_out(tmp_path):
    for manifest in (None, tmp_path / 'negatives.jsonl'):
        cmd = fb.training(tmp_path, tmp_path / 'raw', tmp_path / 'eval',
                          'listed' if manifest else 'b1', tmp_path / 'output', manifest)
        assert cmd[cmd.index('--involve_qa_epochs') + 1] == '3'
        assert cmd[cmd.index('--lora_r') + 1] == '24'
        assert cmd[cmd.index('--s2_optimizer') + 1] == 'inherit_s1'
        assert cmd[cmd.index('--save_steps') + 1] == '10'
        assert cmd[cmd.index('--save_total_limit') + 1] == '2'
        assert 'test' not in cmd and '--official_drop_last_line' not in cmd
        assert '--s2_stop_loss_threshold' not in cmd and '--no_resume' not in cmd


def test_same_negatives_scaled_per_question(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'question_id': 'task_qa:1', 'positive_relation': 'gold',
        'negative_relations': ['a', 'b'], 'hard_negative_relations': ['a'],
        'uniform_negative_relations': ['b'], 'shared_pool': 'train_only_valid_negatives',
        'negative_source': 'shared_valid_pool_calibrated', 'variant': 'calibrated',
        'lambda_candidate': 0.8}) + '\n')
    for (name, method, scale), expected in zip(fb.ARMS, (1.0, 0.8, 0.4)):
        output = tmp_path / (name + '.jsonl')
        fb.hpo.freeze_manifest(source, output, {'method': method, 'lambda_scale': scale})
        row = json.loads(output.read_text())
        assert row['negative_relations'] == ['a', 'b']
        assert row['lambda_candidate'] == expected


def test_validation_gate_rejects_test_and_partial_results(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps({'validation': {'count': 10000, 'official_em': 0.8}}))
    assert fb.full_validation(path) == 0.8
    path.write_text(json.dumps({'validation': {'count': 9999, 'official_em': 0.8}}))
    with pytest.raises(ValueError):
        fb.full_validation(path)
    path.write_text(json.dumps({'validation': {'count': 10000, 'official_em': 0.8}, 'test': {}}))
    with pytest.raises(ValueError):
        fb.full_validation(path)


def test_tasks_and_scoring_use_train_only(tmp_path):
    stages = {n: c for n, c, _ in fb.steps(tmp_path, tmp_path / 'raw', tmp_path / 'data')}
    assert stages['tasks'][stages['tasks'].index('--ref-file') + 1].endswith('processed_train.json')
    assert '--require-train-ref' in stages['tasks']
    assert stages['score'][stages['score'].index('--filter_splits') + 1] == 'train'
    assert '--resume' in stages['score']

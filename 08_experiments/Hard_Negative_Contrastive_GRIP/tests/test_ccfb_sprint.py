"""Sprint seed identity, fixed-primary selection, and verified stage recovery."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import run_ccfb_sprint as sprint


def args(tmp_path, dataset, seed):
    return argparse.Namespace(root=tmp_path / f'{dataset}_{seed}', dataset=dataset,
                              seed=seed, shared_root=None)


def test_paired_seeds_change_both_sampler_and_s2_without_regenerating_s1(tmp_path):
    plan = sprint.build_plan(args(tmp_path, 'fb15k237', 2027))
    assert [name for name, _, _ in plan['stages']] == ['b1', 'freeze']
    freeze = plan['stages'][-1][1]
    assert freeze[freeze.index('--seed') + 1] == '2027'
    for command in plan['training_commands'].values():
        assert command[command.index('--seed') + 1] == '2027'
        assert command[command.index('--involve_qa_epochs') + 1] == '3'
        assert command[command.index('--lora_r') + 1] == '24'
        assert command[command.index('--eval_splits') + 1] == 'validation'
    assert plan['spec']['fixed_main'] == 'soft_mix_1'
    assert 'fb15k237_three_configs_seed2026/tasks/tasks.json' in plan['task']


def test_codex_first_run_uses_new_train_only_task_protocol_and_rank4(tmp_path):
    plan = sprint.build_plan(args(tmp_path, 'codexm', 2026))
    stages = {name: cmd for name, cmd, _ in plan['stages']}
    assert set(stages) == {'tasks', 's1', 'b1', 'score', 'freeze'}
    task = stages['tasks']
    assert task[task.index('--ref-file') + 1].endswith('/codexm/processed_train.json')
    assert '--require-train-ref' in task and '--sample-node-attribute-task' in task
    assert '--no-format-as-instruction' in task and '--train-entity-nodes-only' in task
    assert stages['s1'][stages['s1'].index('--lora_r') + 1] == '4'
    assert stages['score'][stages['score'].index('--filter_splits') + 1] == 'train'
    assert stages['score'][stages['score'].index('--b1_adapter') + 1].endswith('/b1/b1/adapter')


def test_nell_matched_b1_and_existing_primary_guards(tmp_path):
    plan = sprint.build_plan(args(tmp_path, 'nell23k', 2026))
    assert plan['arms'] == ['b1']
    command = plan['training_commands']['b1']
    assert '--relation_text_file' not in command
    assert '--official_drop_last_line' in command
    assert command[command.index('--lambda_candidate') + 1] == '0'
    for dataset in ('fb15k237', 'wn18rr'):
        with pytest.raises(ValueError, match='existing seed2026'):
            sprint.build_plan(args(tmp_path, dataset, 2026))
    with pytest.raises(ValueError, match='outside'):
        sprint.build_plan(args(tmp_path, 'wn18rr', 2031))


def test_nell_harness_uses_source_ids_and_selection_refuses_test():
    rows = [{'split': 'test', 'question_id': f'nell:test:{i}', 'official_correct': i != 1}
            for i in (2, 0, 1)]
    assert sprint.nell_harness(rows, split='test', expected=3) == {
        'official_harness_count': 2, 'official_harness_em': .5}
    with pytest.raises(ValueError, match='complete'):
        sprint.nell_harness(rows[:-1], split='test', expected=3)
    plan = {'dataset': 'fb15k237', 'spec': {'validation_count': 2}}
    with pytest.raises(ValueError, match='validation only'):
        sprint.validation({'validation': {'count': 2, 'official_em': 1}, 'test': {}}, plan)


def test_teacher_with_heldout_filter_or_different_tasks_is_rejected(tmp_path):
    score = tmp_path / 'confusion'; score.mkdir()
    meta = {'filter_splits': ['train'], 'split': 'train', 'dataset': str(tmp_path / 'tasks.json'), 'number_of_qa': 2}
    (score / 'metadata.json').write_text(json.dumps(meta))
    (score / 'run_stats.json').write_text(json.dumps({'qa_count': 2}))
    plan = {'score': str(score), 'task': str(tmp_path / 'tasks.json')}
    sprint.verify_teacher(plan)
    meta['filter_splits'] = ['train', 'test']
    (score / 'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='train-only'):
        sprint.verify_teacher(plan)
    meta['filter_splits'] = ['train']; meta['dataset'] = str(tmp_path / 'old_tasks.json')
    (score / 'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='different task'):
        sprint.verify_teacher(plan)


def test_stage_resume_checks_hashes_and_test_waits_for_frozen_selection(tmp_path, monkeypatch):
    # Exercise orchestration with explicit fake model artifacts, never a GPU run.
    primary = tmp_path / 'primary'; primary.mkdir()
    base = tmp_path / 'project'; (base / 'results/runs').mkdir(parents=True)
    monkeypatch.setattr(sprint, 'HNG', base)
    monkeypatch.setattr(sprint, 'CODE', tmp_path)
    monkeypatch.setattr(sprint, 'verify_data', lambda plan: None)
    monkeypatch.setattr(sprint, 'check_idle_gpu', lambda: None)
    monkeypatch.setattr(sprint.shutil, 'disk_usage', lambda _: SimpleNamespace(free=200 * 1024**3))
    monkeypatch.setenv('TMUX', 'unit-test')
    a = args(tmp_path, 'nell23k', 2026); root = a.root
    command = ['fake-trainer', '--stage', 'b1', '--output_dir', str(root / 'b1')]
    adapter = root / 'b1/b1/adapter/adapter_model.safetensors'
    summary = root / 'b1/b1/summary.json'
    plan = {'dataset': 'nell23k', 'initial': False, 'primary': str(primary), 'arms': ['b1'],
        'missing_dependencies': [], 'missing_sources': [], 'evaluation': 'frozen-eval',
        'spec': {'validation_count': 4951, 'test_count': 4944, 'fixed_main': 'soft_mix_1'},
        'stages': [('b1', command, [str(adapter), str(summary)])]}
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == 'fake-trainer':
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b'fake-adapter')
            summary.write_text(json.dumps({'validation': {'count': 4951,
                'official_harness_count': 4950, 'official_harness_em': .8}}))
        else:
            assert json.loads((root / 'selection.json').read_text())['test_seen'] is False
            output = Path(cmd[cmd.index('--output_dir') + 1]); output.mkdir(parents=True)
            (output / 'summary.json').write_text(json.dumps({'test': {'count': 4944, 'official_em': 1.0}}))
            (output / 'predictions_correct.jsonl').write_text(''.join(json.dumps({
                'split': 'test', 'question_id': f'nell:test:{i}', 'official_correct': True}) + '\n' for i in range(4944)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(sprint.subprocess, 'run', fake_run)
    sprint.execute(a, plan)
    assert len(calls) == 2
    assert json.loads((root / 'comparison.json').read_text())['rows'][0]['official_harness_count'] == 4943
    sprint.execute(a, plan)
    assert len(calls) == 2  # verified outputs reused
    adapter.write_bytes(b'changed-adapter')
    with pytest.raises(ValueError, match='Completed stage changed'):
        sprint.execute(a, plan)
    assert len(calls) == 2

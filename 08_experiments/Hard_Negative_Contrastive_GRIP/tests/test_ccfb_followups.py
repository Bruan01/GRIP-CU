"""CPU preparation preserves live runs and budgets retained checkpoints."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import prepare_ccfb_followups as prep


def test_remaining_matrix_has_41_followup_s2_and_does_not_duplicate_primaries():
    specs = prep.followup_specs()
    assert len(specs) == 11
    assert sum(1 if x == ('nell23k', 2026) else 4 for x in specs) == 41
    assert all((dataset, 2026) not in specs for dataset in ('fb15k237', 'wn18rr', 'codexm'))


def test_retention_budget_counts_two_checkpoints_per_arm_and_new_s1():
    units = {4: {'final': 20, 'checkpoint': 30}, 24: {'final': 40, 'checkpoint': 50}}
    b = prep.disk_budget({4: 21, 24: 24}, units, new_s1_bytes=25, reserve_gib=10)
    assert b['final_adapter_bytes'] == 1380
    assert b['retained_s2_checkpoint_bytes'] == 3660
    assert b['new_shared_s1_bytes'] == 85
    assert b['required_free_bytes'] == 5125 + 513 + 10 * prep.GIB
    archived = prep.disk_budget({4: 21, 24: 24}, units, new_s1_bytes=25,
                                keep_checkpoints=False, reserve_gib=15)
    assert archived['retained_s2_checkpoint_bytes'] == 0
    assert archived['new_shared_s1_bytes'] == 25
    assert archived['transient_checkpoint_bytes'] == 150


@pytest.mark.parametrize('artifact', ['experiment_plan.json', 'run_status.json',
                                    'trainer/checkpoint-1/state', 'b1/adapter_model.safetensors'])
def test_preparation_refuses_started_runs_without_changing_them(tmp_path, artifact):
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'original')
    with pytest.raises(ValueError, match='started experiment'):
        with prep.preparation_lock(tmp_path):
            pytest.fail('must refuse before materializing data')
    assert path.read_bytes() == b'original'


def test_launch_rejects_insufficient_disk_even_when_original_12gib_gate_would_pass(monkeypatch):
    row = {'status': 'inputs_ready', 'spec': {'lora_r': 4}, 'arms': ['b1']}
    units = {4: {'final': 2 * prep.GIB, 'checkpoint': 2 * prep.GIB}}
    monkeypatch.setattr(prep.sprint, 'check_idle_gpu', lambda: pytest.fail('disk must be checked first'))
    with pytest.raises(RuntimeError, match='needs 16.60 GiB'):
        prep.launch_check(row, units, {'disk_bytes': {'free': 13 * prep.GIB}})
    row['status'] = 'waiting_for_primary'
    with pytest.raises(RuntimeError, match='not ready'):
        prep.launch_check(row, units, {'disk_bytes': {'free': 200 * prep.GIB}})


def test_four_arm_launch_retains_original_45gib_minimum_and_checks_busy_gpu(monkeypatch):
    row = {'status': 'inputs_ready', 'spec': {'lora_r': 4}, 'arms': ['b1', 'a', 'b', 'c']}
    units = {4: {'final': prep.GIB, 'checkpoint': prep.GIB}}
    with pytest.raises(RuntimeError, match='needs 45.00 GiB'):
        prep.launch_check(row, units, {'disk_bytes': {'free': 44 * prep.GIB}})
    def busy():
        raise RuntimeError('GPU busy')
    monkeypatch.setattr(prep.sprint, 'check_idle_gpu', busy)
    with pytest.raises(RuntimeError, match='GPU busy'):
        prep.launch_check(row, units, {'disk_bytes': {'free': 50 * prep.GIB}})


def test_sampler_receipt_accepts_equivalent_python_paths_but_distinguishes_venvs(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    for name in ('env-a', 'env-b'):
        bin_dir = tmp_path / name / 'bin'
        bin_dir.mkdir(parents=True)
        (bin_dir / 'python').symlink_to(sys.executable)

    def identity(python, seed_arg='2027'):
        return prep.sampler_identity({'seed': 2027, 'input_sha256': {}, 'source_sha256': {},
            'training_commands': {'b1': [str(python), 'train.py', '--seed', seed_arg]}})

    canonical = tmp_path / 'env-a/bin/python'
    assert identity(project / '../env-a/bin/python') == identity(canonical)
    assert identity(tmp_path / 'env-b/bin/python') != identity(canonical)
    assert identity(canonical, '2028') != identity(canonical)

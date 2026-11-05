#!/usr/bin/env python3
"""CPU preparation and measured disk budgets for the remaining fixed S2 matrix.

Does not train, load a model, start tmux, delete checkpoints, or change a live
experiment. --prepare_negatives materializes the already specified samplers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess

import run_ccfb_sprint as sprint

HNG, CODE, hpo = sprint.HNG, sprint.CODE, sprint.hpo
GIB = 1024**3
PREPARATION = HNG / 'results/runs/ccfb_preparation_20261104/followups'


def followup_specs():
    matrix = hpo.read_json(sprint.MATRIX)['datasets']
    return [(dataset, seed) for dataset, spec in matrix.items() for seed in spec['seeds']
            if seed != 2026 or dataset == 'nell23k']


def allocated_bytes(path):
    return int(subprocess.check_output(['du', '-sx', '-B1', str(path)],
                                      text=True, encoding='utf-8').split()[0])


def memory_and_disk():
    memory = {}
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key] = int(value.split()[0]) * 1024
    gpu = subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader'], text=True, encoding='utf-8').strip()
    return {'memory_bytes': memory, 'disk_bytes': shutil.disk_usage(HNG)._asdict(), 'gpu': gpu}


def measure_model_units():
    runs = HNG / 'results/runs'
    units, evidence = {}, {}
    for rank, adapter, checkpoint, optimizer in [
        (4, runs / 'paper_v2_nell23k/b1_seed2026/b1/adapter',
         runs / 'paper_nell23k/trainer_s1/checkpoint-159',
         runs / 'paper_v2_nell23k/s1_adapter/s1_optimizer.pt'),
        (24, runs / 'wn18rr_three_configs_seed2026/b1_seed2026/b1/adapter',
         runs / 'wn18rr_three_configs_seed2026/b1_seed2026/trainer_b1/checkpoint-117',
         runs / 'wn18rr_three_configs_seed2026/s1_adapter/s1_optimizer.pt'),
    ]:
        final = allocated_bytes(adapter)
        # Remains usable after an explicitly authorized checkpoint archive.
        checkpoint_size = (allocated_bytes(checkpoint) if checkpoint.exists()
                           else final + allocated_bytes(optimizer) + 2 * 1024**2)
        units[rank] = {'final': final, 'checkpoint': checkpoint_size}
        evidence[str(rank)] = {'final_path': str(adapter), 'final_bytes': final,
            'checkpoint_path': str(checkpoint), 'checkpoint_bytes': checkpoint_size,
            'checkpoint_measurement': 'du' if checkpoint.exists() else 'adapter + optimizer + 2 MiB'}
    s1 = allocated_bytes(runs / 'paper_v2_nell23k/s1_adapter')
    return units, s1, evidence


def disk_budget(counts, units, new_s1_bytes=0, *, keep_checkpoints=True, reserve_gib=10):
    final = sum(count * units[rank]['final'] for rank, count in counts.items())
    checkpoints = (2 * sum(count * units[rank]['checkpoint'] for rank, count in counts.items())
                   if keep_checkpoints else 0)
    shared = new_s1_bytes + (2 * units[4]['checkpoint']
                            if new_s1_bytes and keep_checkpoints else 0)
    temporary = (0 if keep_checkpoints else
                 3 * max(units[rank]['checkpoint'] for rank, n in counts.items() if n))
    subtotal = final + checkpoints + shared + temporary
    allowance = (subtotal + 9) // 10
    required = subtotal + allowance + reserve_gib * GIB
    return {'s2_by_lora_rank': {str(k): v for k, v in counts.items()},
        'final_adapter_bytes': final, 'retained_s2_checkpoint_bytes': checkpoints,
        'new_shared_s1_bytes': shared, 'transient_checkpoint_bytes': temporary,
        'allowance_10pct_bytes': allowance, 'reserve_bytes': reserve_gib * GIB,
        'required_free_bytes': required, 'required_free_gib': required / GIB}


@contextmanager
def preparation_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'pipeline.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ((root / 'experiment_plan.json').exists() or (root / 'run_status.json').exists()
                or any(root.rglob('checkpoint-*')) or any(root.rglob('adapter_model.safetensors'))):
            raise ValueError('Preparation refuses a started experiment: ' + str(root))
        yield


def prepare_negatives(plan, output):
    if plan['arms'] == ['b1']:
        return {'status': 'not_needed_for_b1'}
    root = Path(plan['root'])
    with preparation_lock(root):
        receipt = root / 'sampler_preparation.json'
        identity = {key: plan[key] for key in ('input_sha256', 'source_sha256', 'seed')}
        identity['training_commands'] = plan['training_commands']
        if receipt.exists():
            old = hpo.read_json(receipt)
            if old['identity'] != identity:
                raise ValueError('Prepared sampler inputs changed: ' + str(receipt))
            if old['status'] == 'MATERIALIZED':
                if any(hpo.sha256(path) != sha for path, sha in old['outputs'].items()):
                    raise ValueError('Prepared sampler output changed: ' + str(receipt))
                return {'status': 'verified_existing', 'receipt': str(receipt),
                        'outputs': old['outputs'], 'wiring': old['wiring']}
        elif (root / 'sampler').exists() or any(root.glob('*/negatives.jsonl')):
            raise ValueError('Unowned sampler files already exist: ' + str(root))
        # Ownership permits resuming this exact CPU preparation after interruption.
        hpo.write_json(receipt, {'status': 'PREPARING', 'identity': identity})
        freeze = next(cmd for name, cmd, _ in plan['stages'] if name == 'freeze')
        with (output / f"{plan['dataset']}_seed{plan['seed']}_sampler.log").open('a', encoding='utf-8') as log:
            subprocess.run(freeze, check=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
        from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
        expected = hpo.read_json(Path(plan['score']) / 'run_stats.json')['qa_count']
        wiring = {}
        for arm, method, scale in sprint.fb.ARMS:
            manifest = root / arm / 'negatives.jsonl'
            hpo.freeze_manifest(root / 'sampler/calibrated.jsonl', manifest,
                                {'method': method, 'lambda_scale': scale})
            result = verify_listed_manifest_wiring(task_path=Path(plan['task']),
                manifest_paths={method: manifest}, raw_dir=Path(plan['raw']),
                expected_listed=expected, variants=(method,),
                relation_text_file=None if plan['dataset'] == 'nell23k'
                else Path(plan['raw']) / 'rel2text.json')
            wiring[arm] = {k: v for k, v in result['variants'][method].items()
                           if k not in ('question_ids', 'negatives_by_id')}
        files = sorted(p for p in (root / 'sampler').rglob('*') if p.is_file())
        files += [root / arm / 'negatives.jsonl' for arm, _, _ in sprint.fb.ARMS]
        outputs = {str(p): hpo.sha256(p) for p in files}
        hpo.write_json(receipt, {'status': 'MATERIALIZED', 'identity': identity,
            'outputs': outputs, 'wiring': wiring})
        return {'status': 'materialized', 'receipt': str(receipt), 'outputs': outputs, 'wiring': wiring}


def inspect_job(dataset, seed, output, *, materialize=False):
    args = argparse.Namespace(dataset=dataset, seed=seed, shared_root=None,
        root=HNG / 'results/runs/ccfb' / dataset / f'seed{seed}')
    plan = sprint.build_plan(args)
    path = output / 'plans' / f'{dataset}_seed{seed}.json'
    hpo.write_json(path, plan)
    row = {key: plan[key] for key in ('dataset', 'seed', 'root', 'arms', 'spec',
                                     'missing_dependencies', 'missing_sources')}
    row['preview_plan'] = str(path)
    row['preview_plan_sha256'] = hpo.sha256(path)
    if plan['missing_sources']:
        row['status'] = 'blocked_missing_source'
    elif plan['missing_dependencies']:
        primary = Path(plan['primary'])
        if dataset == 'codexm' and all(str(primary) in item for item in plan['missing_dependencies']):
            # Validate static data independently; actual primary assets do not exist yet.
            initial = sprint.build_plan(argparse.Namespace(dataset=dataset, seed=2026,
                shared_root=None, root=primary))
            sprint.verify_data(initial)
            row['status'] = 'waiting_for_primary'
            row['static_data_validation'] = 'PASS'
        else:
            row['status'] = 'blocked_missing_input'
    else:
        sprint.verify_data(plan)
        row['status'] = 'inputs_ready'
        if materialize:
            row['negative_preparation'] = prepare_negatives(plan, output)
    return row


def launch_check(row, units, resources):
    if row['status'] != 'inputs_ready':
        raise RuntimeError('Inputs are not ready: ' + row['status'])
    budget = disk_budget({row['spec']['lora_r']: len(row['arms'])}, units)
    minimum = max(budget['required_free_bytes'], (12 if row['arms'] == ['b1'] else 45) * GIB)
    budget['launch_minimum_free_bytes'] = minimum
    if resources['disk_bytes']['free'] < minimum:
        raise RuntimeError(f"This batch needs {minimum / GIB:.2f} GiB free "
                           'including its retained checkpoints and 10 GiB reserve')
    sprint.check_idle_gpu()
    return budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=PREPARATION)
    parser.add_argument('--dataset', choices=list(hpo.read_json(sprint.MATRIX)['datasets']))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--prepare_negatives', action='store_true')
    parser.add_argument('--check_launch', action='store_true')
    args = parser.parse_args()
    if bool(args.dataset) != (args.seed is not None):
        parser.error('--dataset and --seed must be supplied together')
    if args.check_launch and (args.dataset is None or args.prepare_negatives):
        parser.error('--check_launch requires one dataset/seed and does not prepare samplers')
    specs = followup_specs()
    if args.dataset:
        if (args.dataset, args.seed) not in specs:
            parser.error('This is not a follow-up batch; leave primary entry points unchanged')
        specs = [(args.dataset, args.seed)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset, seed in specs:
        print(f'[prepare] checking {dataset} seed{seed}', flush=True)
        try:
            jobs.append(inspect_job(dataset, seed, args.output_dir, materialize=args.prepare_negatives))
        except Exception as error:
            jobs.append({'dataset': dataset, 'seed': seed, 'status': 'validation_failed',
                         'error': type(error).__name__ + ': ' + str(error)})
    resources = memory_and_disk()
    units, s1, evidence = measure_model_units()
    for row in jobs:
        if 'spec' in row:
            budget = disk_budget({row['spec']['lora_r']: len(row['arms'])}, units)
            row['disk_budget'] = budget
            row['launch_minimum_free_bytes'] = max(budget['required_free_bytes'],
                (12 if row['arms'] == ['b1'] else 45) * GIB)
            row['disk_ready_at_observation'] = (resources['disk_bytes']['free']
                                                >= row['launch_minimum_free_bytes'])
    counts = Counter()
    for dataset, seed in followup_specs():
        spec = hpo.read_json(sprint.MATRIX)['datasets'][dataset]
        counts[spec['lora_r']] += 1 if (dataset, seed) == ('nell23k', 2026) else 4
    primary = HNG / 'results/runs/codexm_three_configs_seed2026'
    current = hpo.read_json(primary / 'run_status.json') if (primary / 'run_status.json').exists() else {}
    if current.get('status') != 'COMPLETE':
        counts[4] += 4  # Conservatively include every arm of the current first run.
    s1_new = s1 if not (primary / 's1_adapter/adapter_model.safetensors').exists() else 0
    frozen = primary / 'experiment_plan.json'
    mismatches = [p for p, sha in hpo.read_json(frozen)['source_sha256'].items()
                  if hpo.sha256(p) != sha] if frozen.exists() else []
    report = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'resources': resources,
        'preparation_script_sha256': hpo.sha256(Path(__file__)),
        'matrix_sha256': hpo.sha256(sprint.MATRIX),
        'current_codexm_source_mismatches': mismatches,
        'jobs': jobs, 'current_codexm': current, 'measurement_evidence': evidence,
        'budget_scope': 'Current CoDEx-M initial run plus 11 follow-up batches; '
                        'follow-ups are conservatively counted even if subsequently completed',
        'retention_2_checkpoints': disk_budget(counts, units, s1_new, reserve_gib=15),
        'archive_completed_checkpoints_externally': disk_budget(counts, units, s1_new,
                                                               keep_checkpoints=False, reserve_gib=15)}
    if current.get('status') != 'COMPLETE':
        report['current_codexm_estimated_remaining'] = disk_budget({4: 4}, units, s1_new)
    if args.check_launch:
        report['launch_budget'] = launch_check(jobs[0], units, resources)
    name = f'launch_check_{args.dataset}_seed{args.seed}.json' if args.check_launch else 'report.json'
    hpo.write_json(args.output_dir / name, report)
    print(json.dumps({'report': str(args.output_dir / name),
        'jobs': [{k: v for k, v in row.items() if k in ('dataset', 'seed', 'status', 'error')}
                 for row in jobs], 'disk_free_gib': resources['disk_bytes']['free'] / GIB,
        'retention_required_gib': report['retention_2_checkpoints']['required_free_gib'],
        'archive_required_gib': report['archive_completed_checkpoints_externally']['required_free_gib']},
        ensure_ascii=False), flush=True)
    if mismatches or any(row['status'].startswith('blocked') or row['status'] == 'validation_failed'
                         for row in jobs):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

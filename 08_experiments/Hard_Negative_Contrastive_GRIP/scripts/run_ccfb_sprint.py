#!/usr/bin/env python3
"""CoDEx-M first run, NELL matched B1, and fixed-primary paired S2 repeats.

Dry-run is CPU-only and reports missing prerequisites. Actual runs require
tmux, an idle GPU, verified inputs, locks and free disk. Existing FB/WN primary
entry points remain the owners of their seed2026 results.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

import run_fb_three_configs as fb

HNG, CODE, hpo = fb.HNG, fb.CODE, fb.hpo
MATRIX = HNG / 'configs/ccfb_sprint_matrix.json'
ARMS = ['b1'] + [name for name, _, _ in fb.ARMS]


def option(command, name, value=None, *, remove=False):
    command = list(command)
    if name in command:
        i = command.index(name)
        if remove:
            del command[i:i + 2]
        else:
            command[i + 1] = str(value)
    elif not remove:
        command += [name, str(value)]
    return command


def paths(dataset, shared_root):
    raw = (CODE if dataset == 'nell23k' else HNG) / 'data/raw_datasets' / dataset
    data = HNG / 'data' / dataset
    evaluation = data / ('recurrent_relation_prediction_full.aligned.json'
                          if dataset == 'nell23k' else 'recurrent_relation_prediction_full.json')
    task = shared_root / 'tasks' / ('grip_nell23k_paper_tasks.json' if dataset == 'nell23k' else 'tasks.json')
    return raw, data, evaluation, task, shared_root / 's1_adapter'


def train_command(args, spec, task, s1, raw, evaluation, arm):
    stage = 'b1' if arm == 'b1' else 'listed'
    manifest = None if arm == 'b1' else args.root / arm / 'negatives.jsonl'
    cmd = fb.training(args.root, raw, evaluation, stage, args.root / arm, manifest)
    for flag, value in [('--seed', args.seed), ('--input_file', task), ('--s1_adapter', s1),
                        ('--lora_r', spec['lora_r']), ('--lora_alpha', spec['lora_alpha'])]:
        cmd = option(cmd, flag, value)
    if args.dataset == 'nell23k':
        cmd = option(cmd, '--relation_text_file', remove=True)
        cmd.append('--official_drop_last_line')
    return cmd


def build_plan(args):
    spec = hpo.read_json(MATRIX)['datasets'][args.dataset]
    if args.seed not in spec['seeds']:
        raise ValueError('Seed is outside the sprint matrix')
    initial = args.dataset == 'codexm' and args.seed == 2026
    primary = HNG / 'results/runs' / spec['primary_root']
    if args.shared_root is not None:
        primary = args.shared_root
    if initial and args.shared_root is not None:
        raise ValueError('CoDEx-M seed2026 creates its own tasks/S1; no shared_root override')
    if args.dataset in ('fb15k237', 'wn18rr') and args.seed == 2026:
        raise ValueError('Resume the existing seed2026 entry point; do not duplicate its four arms')
    selected_arms = ['b1'] if args.dataset == 'nell23k' and args.seed == 2026 else ARMS
    raw, data, evaluation, task, s1 = paths(args.dataset, args.root if initial else primary)
    score = (args.root if initial else primary) / 'confusion'
    stages = []
    required = [raw / name for name in ('train.txt', 'valid.txt', 'test.txt')] + [evaluation, MATRIX]
    if args.dataset != 'nell23k':
        required += [data / 'preparation_manifest.json', raw / 'rel2text.json']
        required += [raw / ('entity2wikidata.json' if args.dataset == 'fb15k237' else 'entity2text.txt')]
    if initial:
        for name, cmd, outputs in fb.steps(args.root, raw, data):
            if name not in ('tasks', 's1'):
                continue
            if name == 's1':
                cmd = option(option(cmd, '--lora_r', spec['lora_r']), '--lora_alpha', spec['lora_alpha'])
            stages.append((name, cmd, list(map(str, outputs))))
        required.append(data / 'processed_train.json')
    else:
        required += [task, s1 / 'adapter_config.json', s1 / 'adapter_model.safetensors', s1 / 's1_optimizer.pt']
        if len(selected_arms) > 1:
            required += [score / name for name in ('candidate_scores.jsonl', 'metadata.json', 'run_stats.json')]
    commands = {arm: train_command(args, spec, task, s1, raw, evaluation, arm) for arm in selected_arms}
    stages.append(('b1', commands['b1'], [str(args.root / 'b1/b1/adapter/adapter_model.safetensors'),
                                        str(args.root / 'b1/b1/summary.json')]))
    if initial:
        # Reuse the inspected offline scorer, changing only paths to the new B1.
        for name, cmd, outputs in fb.steps(args.root, raw, data):
            if name == 'score':
                cmd = option(cmd, '--b1_adapter', args.root / 'b1/b1/adapter')
                stages.append((name, cmd, list(map(str, outputs))))
    if len(selected_arms) > 1:
        sampler = args.root / 'sampler'
        freeze = fb.command(HNG / 'scripts/freeze_shared_pool_samplers.py',
            '--scores', score / 'candidate_scores.jsonl', '--metadata', score / 'metadata.json',
            '--output_dir', sampler, '--k_fixed', 9, '--k_min', 1, '--k_max', 20,
            '--k_uniform', 6, '--k_soft', 3, '--soft_pool_k', 9, '--rho', .33,
            '--lambda_0', 1, '--lambda_min', .25, '--lambda_beta', .5,
            '--seed', args.seed, '--variants', 'calibrated')
        stages.append(('freeze', freeze, [str(sampler / 'calibrated.jsonl')]))
    missing = [str(path) for path in required if not path.is_file()]
    if not initial and args.dataset in ('wn18rr', 'codexm'):
        status = primary / 'run_status.json'
        if not status.exists() or hpo.read_json(status).get('status') != 'COMPLETE':
            missing.append('primary experiment must be COMPLETE: ' + str(primary))
        if not (primary / 'selection.json').is_file():
            missing.append(str(primary / 'selection.json'))
        else:
            required.append(primary / 'selection.json')
    sources = list((HNG / 'src/hard_negative_grip').glob('*.py')) + [Path(__file__), MATRIX,
        Path(fb.__file__), HNG / 'scripts/search_validation_hparams.py',
        HNG / 'scripts/train_listed_contrastive.py', HNG / 'scripts/eval_listed_decode.py',
        HNG / 'scripts/score_relation_candidates_offline.py', HNG / 'scripts/freeze_shared_pool_samplers.py',
        HNG / 'scripts/prepare_codexm.py', HNG / 'scripts/download_codexm.py',
        CODE / 'scripts/generate_nell23k_tasks.py']
    return {'dataset': args.dataset, 'seed': args.seed, 'initial': initial, 'primary': str(primary),
        'root': str(args.root), 'task': str(task), 's1': str(s1), 'raw': str(raw), 'data': str(data),
        'evaluation': str(evaluation), 'score': str(score), 'arms': selected_arms, 'spec': spec,
        's2_epochs': 3, 's2_early_stopping': False,
        'repeat_scope': 'Fixed task, S1 and teacher scores; S2 optimizer and sampler RNG use the declared seed',
        'selection': 'Initial: validation only. Repeats: carry the primary method across every seed, without reselection',
        'stages': stages, 'training_commands': commands, 'missing_dependencies': missing,
        'input_sha256': {str(path): hpo.sha256(path) for path in required if path.is_file()},
        'source_sha256': {str(path): hpo.sha256(path) for path in sources if path.is_file()},
        'missing_sources': [str(path) for path in sources if not path.is_file()]}


def verify_data(plan):
    data, raw = Path(plan['data']), Path(plan['raw'])
    if plan['dataset'] != 'nell23k':
        prep = hpo.read_json(data / 'preparation_manifest.json')
        if plan['dataset'] in ('wn18rr', 'codexm') and not prep.get('official_counts_validated'):
            raise ValueError('Production runs refuse fixture preparation')
        for name, sha in prep['source_sha256'].items():
            if name not in prep.get('unused_source_files', []):
                if hpo.sha256(raw / name) != sha:
                    raise ValueError('Raw source changed: ' + name)
        for parent, records in [(data, prep['outputs']), (raw, prep.get('raw_outputs', {}))]:
            for name, info in records.items():
                if hpo.sha256(parent / name) != info['sha256']:
                    raise ValueError('Prepared input changed: ' + name)
    record = hpo.read_json(plan['evaluation'])
    for split in ('validation', 'test'):
        questions = [q for q in record['recurrent_questions'] if q['split'] == split]
        if len(questions) != plan['spec'][split + '_count']:
            raise ValueError('Incomplete evaluation split: ' + split)
        ids = [q['question_id'] for q in questions]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate evaluation question IDs')
    if not plan['initial']:
        adapter = hpo.read_json(Path(plan['s1']) / 'adapter_config.json')
        if adapter['r'] != plan['spec']['lora_r'] or adapter['lora_alpha'] != plan['spec']['lora_alpha']:
            raise ValueError('Shared S1 LoRA configuration differs from the sprint matrix')
        if len(plan['arms']) > 1:
            verify_teacher(plan)


def verify_teacher(plan):
    score = Path(plan['score'])
    meta = hpo.read_json(score / 'metadata.json')
    stats = hpo.read_json(score / 'run_stats.json')
    if meta.get('filter_splits') != ['train'] or meta.get('split') != 'train':
        raise ValueError('Teacher negatives require train-only filtering')
    if Path(meta['dataset']).resolve() != Path(plan['task']).resolve():
        raise ValueError('Teacher scored a different task file')
    if stats['qa_count'] <= 0 or stats['qa_count'] != meta['number_of_qa']:
        raise ValueError('Teacher score dump is incomplete')


def validation(summary, plan):
    if 'test' in summary or summary['validation']['count'] != plan['spec']['validation_count']:
        raise ValueError('Selection requires full validation only')
    metric = 'official_harness_em' if plan['dataset'] == 'nell23k' else 'official_em'
    if metric == 'official_harness_em' and summary['validation']['official_harness_count'] != 4950:
        raise ValueError('Incorrect NELL validation harness count')
    return summary['validation'][metric]


def nell_harness(rows, *, split, expected):
    rows = [row for row in rows if row['split'] == split]
    indices = [int(row['question_id'].rsplit(':', 1)[-1]) for row in rows]
    if len(rows) != expected or sorted(indices) != list(range(expected)):
        raise ValueError('NELL harness requires the complete ordered source IDs')
    kept = [row for row, index in zip(rows, indices) if index != expected - 1]
    return {'official_harness_count': len(kept),
            'official_harness_em': sum(bool(row['official_correct']) for row in kept) / len(kept)}


def check_idle_gpu():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError('GPU already has a compute process; leave the current experiment running')


def execute(args, plan):
    if plan['missing_dependencies'] or plan['missing_sources']:
        raise ValueError('Missing prerequisites: ' + str(plan['missing_dependencies'] + plan['missing_sources']))
    if not os.environ.get('TMUX'):
        raise RuntimeError('Use configs/run_ccfb_sprint.sh in tmux')
    verify_data(plan)
    args.root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock_paths = [HNG / 'results/runs/ccfb_gpu.lock', args.root / 'pipeline.lock']
        if not plan['initial']:
            lock_paths.append(Path(plan['primary']) / 'pipeline.lock')
        locks = [stack.enter_context(path.open('a')) for path in lock_paths]
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_idle_gpu()
        frozen = args.root / 'experiment_plan.json'
        minimum = 12 if plan['arms'] == ['b1'] else 45
        if not frozen.exists() and shutil.disk_usage(args.root).free < minimum * 1024**3:
            raise RuntimeError(f'New run requires {minimum} GiB free; no model loaded')
        hpo.freeze_text(frozen, json.dumps(plan, sort_keys=True, indent=2) + '\n')
        stage = 'initialization'

        def run(name, cmd, outputs):
            nonlocal stage
            stage = name
            marker = args.root / 'completed' / (name + '.json')
            if marker.exists():
                saved = hpo.read_json(marker)
                if saved['command'] != cmd or any(hpo.sha256(p) != sha for p, sha in saved['outputs'].items()):
                    raise ValueError('Completed stage changed: ' + name)
                print('[ccfb] reuse ' + name, flush=True)
                return
            if shutil.disk_usage(args.root).free < 10 * 1024**3:
                raise RuntimeError('Less than 10 GiB free; checkpoints retained')
            hpo.write_json(args.root / 'run_status.json', {'status': 'RUNNING', 'stage': name})
            effective = list(cmd)
            if '--stage' in cmd:
                variant = cmd[cmd.index('--stage') + 1]
                out = Path(cmd[cmd.index('--output_dir') + 1])
                adapter = out / variant / 'adapter'
                if variant != 's1' and all((adapter / f).is_file() for f in ('adapter_config.json', 'adapter_model.safetensors')):
                    effective.append('--skip_train')
            print('[ccfb] starting ' + name, flush=True)
            with (args.root / (name + '.log')).open('a') as log:
                subprocess.run(effective, check=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                               pass_fds=tuple(lock.fileno() for lock in locks))
            hpo.write_json(marker, {'command': cmd, 'outputs': {str(p): hpo.sha256(p) for p in outputs}})

        try:
            for name, cmd, outputs in plan['stages']:
                if name == 'freeze':
                    verify_teacher(plan)
                run(name, cmd, outputs)
            if len(plan['arms']) > 1:
                from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
                expected = hpo.read_json(Path(plan['score']) / 'run_stats.json')['qa_count']
                if expected <= 0:
                    raise ValueError('No eligible contrastive QA')
                for name, method, scale in fb.ARMS:
                    directory = args.root / name
                    directory.mkdir(exist_ok=True)
                    hpo.freeze_manifest(args.root / 'sampler/calibrated.jsonl', directory / 'negatives.jsonl',
                                        {'method': method, 'lambda_scale': scale})
                    verify_listed_manifest_wiring(task_path=Path(plan['task']),
                        manifest_paths={method: directory / 'negatives.jsonl'}, raw_dir=Path(plan['raw']),
                        expected_listed=expected, variants=(method,),
                        relation_text_file=None if args.dataset == 'nell23k' else Path(plan['raw']) / 'rel2text.json')
                    run(name, plan['training_commands'][name], [directory / 'listed/adapter/adapter_model.safetensors',
                                                               directory / 'listed/summary.json'])
            scores = {arm: validation(hpo.read_json(args.root / arm / ('b1' if arm == 'b1' else 'listed') / 'summary.json'), plan)
                      for arm in plan['arms']}
            if plan['initial']:
                selected = max([name for name, _, _ in fb.ARMS], key=scores.get)
            else:
                selected = plan['spec']['fixed_main'] or hpo.read_json(Path(plan['primary']) / 'selection.json')['selected']
            if selected not in ARMS[1:]:
                raise ValueError('Invalid frozen main method')
            hpo.freeze_text(args.root / 'selection.json', json.dumps({'selected': selected,
                'validation': scores, 'test_seen': False, 'selection_source':
                'current validation' if plan['initial'] else 'primary validation; no per-seed reselection'},
                sort_keys=True, indent=2) + '\n')
            rows = []
            for arm in plan['arms']:
                output = args.root / 'test' / arm
                cmd = fb.command(HNG / 'scripts/eval_listed_decode.py', '--adapter_dir',
                    args.root / arm / ('b1' if arm == 'b1' else 'listed') / 'adapter',
                    '--output_dir', output, '--eval_file', plan['evaluation'], '--splits', 'test',
                    '--model_name', 'qwen-7b', '--model_cache_dir', CODE / 'model_cache',
                    '--decode', 'generate', '--gen_max_length', 1000, '--progress_every', 100)
                predictions = output / 'predictions_correct.jsonl'
                run('test_' + arm, cmd, [output / 'summary.json', predictions])
                summary = hpo.read_json(output / 'summary.json')['test']
                if summary['count'] != plan['spec']['test_count']:
                    raise ValueError('Incomplete test predictions')
                row = {'arm': arm, 'validation': scores[arm], 'test_official_em': summary['official_em'],
                       'test_count': summary['count'], 'selected': arm == selected}
                if args.dataset == 'nell23k':
                    row.update(nell_harness([json.loads(line) for line in predictions.read_text().splitlines() if line.strip()],
                                            split='test', expected=4944))
                rows.append(row)
                hpo.write_json(args.root / 'comparison.json', {'selected': selected, 'seed': args.seed, 'rows': rows})
            hpo.write_json(args.root / 'run_status.json', {'status': 'COMPLETE', 'selected': selected})
        except (Exception, KeyboardInterrupt) as error:
            hpo.write_json(args.root / 'run_status.json', {'status': 'INTERRUPTED', 'stage': stage, 'error': str(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=list(hpo.read_json(MATRIX)['datasets']))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--shared_root', type=Path)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if args.root is None:
        args.root = HNG / 'results/runs' / ('codexm_three_configs_seed2026'
            if args.dataset == 'codexm' and args.seed == 2026 else f'ccfb/{args.dataset}/seed{args.seed}')
    args.root = args.root.resolve()
    if args.shared_root is not None:
        args.shared_root = args.shared_root.resolve()
    plan = build_plan(args)
    if args.dry_run:
        if not plan['missing_dependencies'] and not plan['missing_sources']:
            verify_data(plan)
        print(json.dumps(plan, indent=2))
        return
    execute(args, plan)


if __name__ == '__main__':
    main()

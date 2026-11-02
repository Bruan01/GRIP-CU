#!/usr/bin/env python3
"""WN18RR fixed-three-epoch seed2026 comparison; not the paper's five-run mean."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

import run_fb_three_configs as fb

HNG, CODE, ARMS, hpo = fb.HNG, fb.CODE, fb.ARMS, fb.hpo
COUNTS = {'validation': 3034, 'test': 3134}


def steps(root, raw, data):
    # Identical task/optimizer budgets to the fixed FB comparison. WN-specific
    # relation text and evaluation records are passed explicitly.
    yield from fb.steps(root, raw, data)


def full_validation(path):
    summary = hpo.read_json(path)
    if 'test' in summary or summary['validation']['count'] != COUNTS['validation']:
        raise ValueError(f'Expected full WN validation only: {path}')
    return summary['validation']['official_em']


def plan_for(root):
    raw, data = HNG / 'data/raw_datasets/wn18rr', HNG / 'data/wn18rr'
    prep = hpo.read_json(data / 'preparation_manifest.json')
    if (not prep.get('official_counts_validated') or
            prep['counts']['source_triples'] != {'train': 86835, 'validation': 3034, 'test': 3134}):
        raise ValueError('WN training requires verified full dataset counts, not fixtures')
    for name, digest in prep['source_sha256'].items():
        if hpo.sha256(raw / name) != digest:
            raise ValueError(f'Raw file changed: {name}')
    for name, info in prep.get('raw_outputs', {}).items():
        if hpo.sha256(raw / name) != info['sha256']:
            raise ValueError(f'Derived raw mapping changed: {name}')
    for name, info in prep['outputs'].items():
        if hpo.sha256(data / name) != info['sha256']:
            raise ValueError(f'Prepared file changed: {name}')
    evaluation = hpo.read_json(data / 'recurrent_relation_prediction_full.json')
    for split, count in COUNTS.items():
        records = [q for q in evaluation['recurrent_questions'] if q['split'] == split]
        if len(records) != count or any(len(set(q['candidate_relations'])) != 11 for q in records):
            raise ValueError(f'WN requires complete {split} with all 11 candidates')
    sources = list((HNG / 'src/hard_negative_grip').glob('*.py')) + [Path(__file__),
        Path(fb.__file__), HNG / 'scripts/prepare_wn18rr.py',
        HNG / 'scripts/train_listed_contrastive.py', HNG / 'scripts/search_validation_hparams.py',
        HNG / 'scripts/score_relation_candidates_offline.py', HNG / 'scripts/freeze_shared_pool_samplers.py',
        HNG / 'scripts/eval_listed_decode.py', CODE / 'scripts/generate_nell23k_tasks.py']
    return {'dataset': 'wn18rr', 'seed': 2026, 'arms': ARMS, 's2_epochs': 3,
        's2_early_stopping': False, 'lora_r': 24, 'lora_alpha': 48,
        'evaluation_counts': COUNTS, 'evaluation_candidates': 11,
        'paper_reference_accuracy': 0.6430, 'paper_reference_runs': 5,
        'comparison_limit': 'Single seed; fixed 3 epochs, not paper loss-threshold stopping or five-run mean',
        'negative_policy': '6 uniform + 3 soft from Top9; capped by train-filtered valid pool size',
        'selection': 'validation.official_em; ties use declared arm order; freeze before any test',
        'preparation_sha256': hpo.sha256(data / 'preparation_manifest.json'),
        'relation_text_sha256': hpo.sha256(raw / 'rel2text.json'),
        'sources': {str(p): hpo.sha256(p) for p in sources},
        'initial_steps': [(n, c, [str(p) for p in outputs]) for n, c, outputs in steps(root, raw, data)],
        'training_commands': {name: fb.training(root, raw, data / 'recurrent_relation_prediction_full.json',
            'listed', root / name, root / name / 'negatives.jsonl') for name, _, _ in ARMS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = plan_for(root)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not os.environ.get('TMUX'):
        raise RuntimeError('Use configs/run_wn_three_configs.sh in tmux')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / 'experiment_plan.json').exists() and shutil.disk_usage(root).free < 45 * 1024**3:
            raise RuntimeError('New WN experiment requires 45 GiB free; no training started')
        hpo.freeze_text(root / 'experiment_plan.json', json.dumps(plan, sort_keys=True, indent=2) + '\n')
        stage = 'initialization'
        def run(name, cmd, outputs):
            nonlocal stage
            stage = name
            marker = root / 'completed' / (name + '.json')
            if marker.exists():
                saved = hpo.read_json(marker)
                if saved['command'] != cmd or any(hpo.sha256(Path(p)) != sha for p, sha in saved['outputs'].items()):
                    raise ValueError(f'Completed stage changed: {name}')
                print(f'[wn] reuse {name}', flush=True)
                return
            if shutil.disk_usage(root).free < 10 * 1024**3:
                raise RuntimeError(f'Less than 10 GiB free before {name}; checkpoints retained')
            hpo.write_json(root / 'run_status.json', {'status': 'RUNNING', 'stage': name})
            effective = list(cmd)
            if '--stage' in cmd:
                train_stage = cmd[cmd.index('--stage') + 1]
                output = Path(cmd[cmd.index('--output_dir') + 1])
                adapter = root / 's1_adapter' if train_stage == 's1' else output / train_stage / 'adapter'
                if train_stage != 's1' and all((adapter / p).is_file() for p in ('adapter_config.json', 'adapter_model.safetensors')):
                    effective.append('--skip_train')
            print(f'[wn] starting {name}', flush=True)
            with (root / (name + '.log')).open('a') as log:
                subprocess.run(effective, check=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                               pass_fds=(lock.fileno(),))
            hpo.write_json(marker, {'command': cmd, 'outputs': {str(p): hpo.sha256(Path(p)) for p in outputs}})
        try:
            for name, cmd, outputs in plan['initial_steps']:
                run(name, cmd, outputs)
            from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
            for name, method, scale in ARMS:
                directory = root / name
                directory.mkdir(exist_ok=True)
                hpo.freeze_manifest(root / 'sampler_seed2026/calibrated.jsonl', directory / 'negatives.jsonl',
                                    {'method': method, 'lambda_scale': scale})
                expected = hpo.read_json(root / 'confusion/run_stats.json')['qa_count']
                if expected <= 0:
                    raise ValueError('No matchable WN relation QA; refusing empty contrastive experiment')
                verify_listed_manifest_wiring(task_path=root / 'tasks/tasks.json',
                    manifest_paths={method: directory / 'negatives.jsonl'}, raw_dir=HNG / 'data/raw_datasets/wn18rr',
                    expected_listed=expected, variants=(method,),
                    relation_text_file=HNG / 'data/raw_datasets/wn18rr/rel2text.json')
                run(name, plan['training_commands'][name], [directory / 'listed/adapter/adapter_model.safetensors',
                                                           directory / 'listed/summary.json'])
            scores = {n: full_validation(root / n / 'listed/summary.json') for n, _, _ in ARMS}
            baseline = full_validation(root / 'b1_seed2026/b1/summary.json')
            best = max(scores, key=scores.get)
            hpo.freeze_text(root / 'selection.json', json.dumps({'selected': best, 'validation': scores,
                'b1_validation': baseline, 'metric': 'validation.official_em', 'test_seen': False}, sort_keys=True, indent=2) + '\n')
            rows = []
            for name, relative in [('b1_seed2026', 'b1')] + [(n, 'listed') for n, _, _ in ARMS]:
                output = root / 'test' / name
                run('test_' + name, fb.command(HNG / 'scripts/eval_listed_decode.py',
                    '--adapter_dir', root / name / relative / 'adapter', '--output_dir', output,
                    '--eval_file', HNG / 'data/wn18rr/recurrent_relation_prediction_full.json', '--splits', 'test',
                    '--model_name', 'qwen-7b', '--model_cache_dir', CODE / 'model_cache',
                    '--decode', 'generate', '--gen_max_length', 1000, '--progress_every', 100), [output / 'summary.json'])
                result = hpo.read_json(output / 'summary.json')['test']
                if result['count'] != COUNTS['test']:
                    raise ValueError('Incomplete WN test evaluation')
                rows.append({'arm': name, 'validation': baseline if name == 'b1_seed2026' else scores[name],
                             'test_official_em': result['official_em'], 'selected': name == best})
                hpo.write_json(root / 'comparison.json', {'selected': best, 'rows': rows})
            hpo.write_json(root / 'run_status.json', {'status': 'COMPLETE', 'selected': best})
        except Exception as exc:
            hpo.write_json(root / 'run_status.json', {'status': 'INTERRUPTED', 'stage': stage, 'error': str(exc)})
            raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Fixed FB15k-237 seed-2026 B1 and three-arm transfer experiment."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import search_validation_hparams as hpo

HNG = Path(__file__).resolve().parents[1]
CODE = HNG.parent / 'RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp'
ARMS = [('soft_mix_1', 'soft_mix', 1.0), ('calibrated_1', 'calibrated', 1.0),
        ('calibrated_05', 'calibrated', 0.5)]


def command(script, *args):
    return [sys.executable, str(script), *map(str, args)]


def training(root, raw, evaluation, stage, output, manifest=None):
    cmd = command(HNG / 'scripts/train_listed_contrastive.py',
        '--input_file', root / 'tasks/tasks.json', '--eval_file', evaluation,
        '--eval_splits', 'validation', '--output_dir', output, '--stage', stage,
        '--s1_adapter', root / 's1_adapter', '--model_name', 'qwen-7b',
        '--model_cache_dir', CODE / 'model_cache', '--raw_dir', raw,
        '--relation_text_file', raw / 'rel2text.json', '--lora_r', 24, '--lora_alpha', 48,
        '--target_modules', 'down_proj', 'up_proj', 'gate_proj',
        '--num_train_epochs', 1, '--s1_stop_loss_threshold', 0.15, '--involve_qa_epochs', 3,
        '--qa_loss_mask', 'answer_only', '--per_device_train_batch_size', 1,
        '--gradient_accumulation_steps', 512, '--s1_gradient_checkpointing',
        '--learning_rate', 0.001, '--weight_decay', 0.0001, '--adam_epsilon', 0.0001,
        '--max_grad_norm', 1, '--s2_optimizer', 'inherit_s1', '--gen_max_length', 1000,
        '--save_steps', 10, '--save_total_limit', 2, '--seed', 2026,
        '--lambda_candidate', 1 if manifest else 0, '--temperature', 1, '--memory_size', 0,
        '--listed_negative_source', 'score_hard' if manifest else 'train_graph')
    if manifest:
        cmd += ['--score_hard_manifest', str(manifest)]
    return cmd


def steps(root, raw, data):
    task = root / 'tasks/tasks.json'
    evaluation = data / 'recurrent_relation_prediction_full.json'
    yield 'tasks', command(CODE / 'scripts/generate_nell23k_tasks.py',
        '--input-file', data / 'processed_train.json', '--ref-file', data / 'processed_train.json',
        '--require-train-ref', '--task-file', task, '--task-cache-dir', root / 'tasks/cache',
        '--seed', 2026, '--model-name', 'qwen-7b', '--model-source', 'local',
        '--model-cache-dir', CODE / 'model_cache', '--local-files-only',
        '--task-generator-model-name', 'qwen-7b', '--no-task-generator-use-vllm',
        '--task-generator-batch-size', 1, '--task-gen-max-length', 1000,
        '--num-context-qa', 8000, '--num-reason-qa', 2000, '--num-summarization', 6000,
        '--sample-node-attribute-task', '--no-repharse-context-qa', '--no-context-upsampling',
        '--no-format-as-instruction', '--uniform-sampling', '--rooted-reasoning-subgraph',
        '--random-reasoning-type', '--keep-parallel-edges', '--train-entity-nodes-only',
        '--involve-qa-epochs', 3), [task]
    yield 's1', training(root, raw, evaluation, 's1', root), [
        root / 's1_adapter/adapter_model.safetensors', root / 's1_adapter/s1_optimizer.pt']
    yield 'b1', training(root, raw, evaluation, 'b1', root / 'b1_seed2026'), [
        root / 'b1_seed2026/b1/adapter/adapter_model.safetensors', root / 'b1_seed2026/b1/summary.json']
    score = root / 'confusion'
    yield 'score', command(HNG / 'scripts/score_relation_candidates_offline.py',
        '--task_file', task, '--raw_dir', raw, '--relation_text_file', raw / 'rel2text.json',
        '--b1_adapter', root / 'b1_seed2026/b1/adapter', '--output_dir', score,
        '--model_name', 'qwen-7b', '--model_cache_dir', CODE / 'model_cache',
        '--limit', 0, '--candidate_batch_size', 8, '--temperature', 1,
        '--filter_splits', 'train', '--progress_every', 20, '--resume'), [
            score / 'run_stats.json', score / 'candidate_scores.jsonl', score / 'metadata.json']
    sampler = root / 'sampler_seed2026'
    yield 'freeze', command(HNG / 'scripts/freeze_shared_pool_samplers.py',
        '--scores', score / 'candidate_scores.jsonl', '--metadata', score / 'metadata.json',
        '--output_dir', sampler, '--k_fixed', 9, '--k_min', 1, '--k_max', 20,
        '--k_uniform', 6, '--k_soft', 3, '--soft_pool_k', 9, '--rho', 0.33,
        '--lambda_0', 1, '--lambda_min', 0.25, '--lambda_beta', 0.5,
        '--seed', 2026, '--variants', 'calibrated'), [sampler / 'calibrated.jsonl']


def full_validation(path):
    summary = hpo.read_json(path)
    if 'test' in summary or summary['validation']['count'] != 10000:
        raise ValueError(f'Expected complete validation-only summary: {path}')
    return summary['validation']['official_em']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    raw, data = HNG / 'data/raw_datasets/fb15k237', HNG / 'data/fb15k237'
    prep = hpo.read_json(data / 'preparation_manifest.json')
    for name, sha in prep['source_sha256'].items():
        if name not in prep.get('unused_source_files', []) and hpo.sha256(raw / name) != sha:
            raise ValueError(f'Raw data hash mismatch: {name}')
    for name, info in prep['outputs'].items():
        if hpo.sha256(data / name) != info['sha256']:
            raise ValueError(f'Prepared data hash mismatch: {name}')
    sources = list((HNG / 'src/hard_negative_grip').glob('*.py')) + [Path(__file__),
        HNG / 'scripts/train_listed_contrastive.py', HNG / 'scripts/search_validation_hparams.py',
        HNG / 'scripts/score_relation_candidates_offline.py', HNG / 'scripts/freeze_shared_pool_samplers.py',
        HNG / 'scripts/eval_listed_decode.py', CODE / 'scripts/generate_nell23k_tasks.py']
    plan = {'dataset': 'fb15k237', 'seed': 2026, 'arms': ARMS,
            's2_epochs': 3, 's2_early_stopping': False, 'lora_r': 24, 'lora_alpha': 48,
            'selection': 'validation.official_em; ties use declared arm order',
            'test_policy': 'All four fixed arms after validation selection; no test-based tuning',
            'preparation_sha256': hpo.sha256(data / 'preparation_manifest.json'),
            'sources': {str(p): hpo.sha256(p) for p in sources},
            'initial_steps': [(n, c) for n, c, _ in steps(root, raw, data)],
            'training_commands': {name: training(root, raw, data / 'recurrent_relation_prediction_full.json',
                'listed', root / name, root / name / 'negatives.jsonl') for name, _, _ in ARMS}}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not os.environ.get('TMUX'):
        raise RuntimeError('Use configs/run_fb_three_configs.sh inside tmux')
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((root / 'pipeline.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = root / 'experiment_plan.json'
        if not plan_path.exists() and shutil.disk_usage(root).free < 45 * 1024**3:
            raise RuntimeError('New FB experiment needs at least 45 GiB free')
        hpo.freeze_text(plan_path, json.dumps(plan, sort_keys=True, indent=2) + '\n')
        status = root / 'run_status.json'

        def run(name, cmd, outputs):
            marker = root / 'completed' / (name + '.json')
            if marker.exists():
                saved = hpo.read_json(marker)
                if saved['command'] != cmd or any(hpo.sha256(p) != sha for p, sha in saved['outputs'].items()):
                    raise ValueError(f'Completed stage changed: {name}')
                print(f'[fb] reuse completed stage={name}', flush=True)
                return
            if shutil.disk_usage(root).free < 10 * 1024**3:
                raise RuntimeError(f'Less than 10 GiB free before {name}; stopped with checkpoints retained')
            hpo.write_json(status, {'status': 'RUNNING', 'stage': name})
            print(f'[fb] starting {name}', flush=True)
            effective = list(cmd)
            if '--stage' in cmd and cmd[cmd.index('--stage') + 1] in ('b1', 'listed'):
                stage = cmd[cmd.index('--stage') + 1]
                adapter = Path(cmd[cmd.index('--output_dir') + 1]) / stage / 'adapter'
                if (adapter / 'adapter_config.json').is_file() and (adapter / 'adapter_model.safetensors').is_file():
                    effective.append('--skip_train')
            with (root / (name + '.log')).open('a') as log:
                subprocess.run(effective, check=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                               pass_fds=(lock.fileno(),))
            hpo.write_json(marker, {'command': cmd, 'outputs': {str(p): hpo.sha256(p) for p in outputs}})

        try:
            for name, cmd, outputs in steps(root, raw, data):
                run(name, cmd, outputs)
            for name, method, scale in ARMS:
                directory = root / name
                directory.mkdir(exist_ok=True)
                hpo.freeze_manifest(root / 'sampler_seed2026/calibrated.jsonl', directory / 'negatives.jsonl',
                                    {'method': method, 'lambda_scale': scale})
                # Verify every scored QA is attached before training; all arms share negative identities.
                from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
                expected = hpo.read_json(root / 'confusion/run_stats.json')['qa_count']
                verify_listed_manifest_wiring(task_path=root / 'tasks/tasks.json',
                    manifest_paths={method: directory / 'negatives.jsonl'}, raw_dir=raw,
                    expected_listed=expected, variants=(method,), relation_text_file=raw / 'rel2text.json')
                run(name, plan['training_commands'][name], [directory / 'listed/adapter/adapter_model.safetensors',
                                                            directory / 'listed/summary.json'])
            scores = {name: full_validation(root / name / 'listed/summary.json') for name, _, _ in ARMS}
            baseline = full_validation(root / 'b1_seed2026/b1/summary.json')
            best = max(scores, key=scores.get)
            hpo.freeze_text(root / 'selection.json', json.dumps({'selected': best, 'validation': scores,
                'b1_validation': baseline, 'metric': 'validation.official_em', 'test_seen': False},
                sort_keys=True, indent=2) + '\n')
            rows = []
            for name, relative in [('b1_seed2026', 'b1')] + [(n, 'listed') for n, _, _ in ARMS]:
                output = root / 'test' / name
                run('test_' + name, command(HNG / 'scripts/eval_listed_decode.py',
                    '--adapter_dir', root / name / relative / 'adapter', '--output_dir', output,
                    '--eval_file', data / 'recurrent_relation_prediction_full.json', '--splits', 'test',
                    '--model_name', 'qwen-7b', '--model_cache_dir', CODE / 'model_cache',
                    '--decode', 'generate', '--gen_max_length', 1000, '--progress_every', 100), [output / 'summary.json'])
                summary = hpo.read_json(output / 'summary.json')['test']
                if summary['count'] != 10000:
                    raise ValueError('Incomplete test evaluation')
                rows.append({'arm': name, 'validation': baseline if name == 'b1_seed2026' else scores[name],
                             'test_official_em': summary['official_em'], 'selected': name == best})
                hpo.write_json(root / 'comparison.json', {'selected': best, 'rows': rows})
            hpo.write_json(status, {'status': 'COMPLETE', 'selected': best})
        except Exception as exc:
            hpo.write_json(status, {'status': 'INTERRUPTED', 'error': str(exc)})
            raise


if __name__ == '__main__':
    main()

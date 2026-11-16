#!/usr/bin/env python3
"""Replicate the WN B1/Soft-Mix contrast from a fresh S1 in one tmux run."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_fb_three_configs as fb
import run_wn_three_configs as wn
from analyze_ccfb_seed2026 import analyze, result_rows
from hard_negative_grip.decode_io import load_jsonl
from hard_negative_grip.shared_pool_samplers import verify_listed_manifest_wiring
from run_ccfb_sprint import check_idle_gpu, option, verify_teacher

HNG, CODE, hpo = fb.HNG, fb.CODE, fb.hpo
RUNS = HNG / 'results/runs'
PRIMARY = RUNS / 'wn18rr_three_configs_seed2026'
RAW = HNG / 'data/raw_datasets/wn18rr'
EVAL = HNG / 'data/wn18rr/recurrent_relation_prediction_full.json'
TASK = PRIMARY / 'tasks/tasks.json'
ARMS = {'b1_seed2026': 'b1', 'soft_mix_1': 'listed'}
QA_COUNT, ELIGIBLE, EXCLUDED, S2_STEPS = 19782, 2697, 79, 117
GIB = 1024**3
RESERVE = 10 * GIB


def allocated_bytes(path):
    return int(subprocess.check_output(['du', '-s', '-B1', str(path)], text=True).split()[0])


def build_plan(root, s1_seed):
    original = wn.plan_for(root)
    if hpo.read_json(PRIMARY / 'run_status.json')['status'] != 'COMPLETE':
        raise ValueError('The original WN pair must be complete')
    if hpo.read_json(PRIMARY / 'run_config.json')['seed'] == s1_seed:
        raise ValueError('S1 seed must differ from the original run')
    task_sha = hpo.sha256(TASK)
    if hpo.read_json(PRIMARY / 'completed/tasks.json')['outputs'][str(TASK)] != task_sha:
        raise ValueError('Original WN task file changed')
    eval_records = hpo.read_json(EVAL)['recurrent_questions']
    for split, count in wn.COUNTS.items():
        if len({row['question_id'] for row in eval_records if row['split'] == split}) != count:
            raise ValueError(f'Expected {count} unique WN {split} IDs')

    model = CODE / 'model_cache/Qwen--Qwen2.5-7B-Instruct'
    index = model / 'model.safetensors.index.json'
    shards = sorted(set(hpo.read_json(index)['weight_map'].values()))
    sizes = {name: (model / name).stat().st_size for name in shards}
    if not sizes or min(sizes.values()) <= 0:
        raise ValueError('Incomplete local Qwen weights')

    # Use the existing full WN recipe. Only S1 changes seed and all three
    # training stages retain one rolling checkpoint to fit the current disk.
    stages = {}
    for name, cmd, outputs in original['initial_steps']:
        if name == 'tasks':
            continue
        if name in ('s1', 'b1'):
            cmd = option(option(cmd, '--input_file', TASK), '--save_total_limit', 1)
        if name == 's1':
            cmd = option(cmd, '--seed', s1_seed)
            outputs += [str(root / 's1_adapter' / leaf) for leaf in ('adapter_config.json', 'run_metadata.json')]
        if name == 'score':
            cmd = option(cmd, '--task_file', TASK)
            outputs.append(str(root / 'confusion/qa_eligibility.json'))
        if name == 'freeze':
            outputs.append(str(root / 'sampler_seed2026/policy.json'))
        stages[name] = [cmd, outputs]
    cmd = option(original['training_commands']['soft_mix_1'], '--input_file', TASK)
    stages['soft_mix_1'] = [option(cmd, '--save_total_limit', 1), []]
    for arm, variant in ARMS.items():
        output = root / arm / variant
        test = root / 'test' / arm
        stages['b1' if variant == 'b1' else arm][1] = [str(output / name) for name in (
            'adapter/adapter_model.safetensors', 'adapter/adapter_config.json',
            'adapter/run_metadata.json', 'summary.json', 'predictions_correct.jsonl')]
        stages['test_' + arm] = [fb.command(HNG / 'scripts/eval_listed_decode.py',
            '--adapter_dir', output / 'adapter', '--output_dir', test, '--eval_file', EVAL,
            '--splits', 'test', '--model_name', 'qwen-7b', '--model_cache_dir', CODE / 'model_cache',
            '--decode', 'generate', '--gen_max_length', 1000, '--progress_every', 100),
            [str(test / name) for name in ('summary.json', 'predictions_correct.jsonl')]]

    s1 = PRIMARY / 's1_adapter'
    exports = [allocated_bytes(s1)] + [allocated_bytes(PRIMARY / arm / variant / 'adapter')
                                      for arm, variant in ARMS.items()]
    checkpoint = max(exports[0], max(exports[1:]) + allocated_bytes(s1 / 's1_optimizer.pt')) + 4 * 1024**2
    estimated = sum(exports) + 4 * checkpoint  # three rolling copies plus one save in progress
    required = (estimated * 11 + 9) // 10 + RESERVE
    sources = original['sources'].copy()
    for path in (Path(__file__), HNG / 'scripts/analyze_ccfb_seed2026.py', HNG / 'scripts/run_ccfb_sprint.py'):
        sources[str(path.resolve())] = hpo.sha256(path)
    return {
        'dataset': 'wn18rr', 'model': model.name, 's1_seed': s1_seed, 's2_seed': 2026,
        'sampler_seed': 2026, 's2_steps': S2_STEPS, 'eval_counts': wn.COUNTS,
        'checkpoint_policy': 'save every 10 steps; retain 1 rolling checkpoint per training stage',
        'test_policy': 'evaluate both fixed final S2 adapters after full validation',
        'original_s1_sha256': hpo.read_json(PRIMARY / 'completed/s1.json')['outputs'][str(s1 / 'adapter_model.safetensors')],
        'input_sha256': {str(TASK): task_sha, str(EVAL): hpo.sha256(EVAL),
                         str(RAW / 'train.txt'): hpo.sha256(RAW / 'train.txt'),
                         str(index): hpo.sha256(index), str(model / 'tokenizer.json'): hpo.sha256(model / 'tokenizer.json')},
        'model_shard_sizes': sizes, 'source_sha256': sources,
        'estimated_peak_additional_bytes': estimated, 'required_free_bytes': required, 'stages': stages,
    }


def verify_soft_mix(root):
    score = root / 'confusion'
    verify_teacher({'score': score, 'task': TASK})
    eligibility = hpo.read_json(score / 'qa_eligibility.json')
    original = hpo.read_json(PRIMARY / 'confusion/qa_eligibility.json')
    teacher = hpo.read_json(score / 'metadata.json')
    if ((eligibility['eligible_before_sharding'], eligibility['excluded_count']) != (ELIGIBLE, EXCLUDED)
            or eligibility != original or Path(teacher['checkpoint']) != root / 'b1_seed2026/b1/adapter'):
        raise ValueError('New B1 teacher or WN eligibility differs from the frozen protocol')
    manifest = root / 'soft_mix_1/negatives.jsonl'
    hpo.freeze_manifest(root / 'sampler_seed2026/calibrated.jsonl', manifest,
                        {'method': 'soft_mix', 'lambda_scale': 1.0})
    report = verify_listed_manifest_wiring(task_path=TASK, manifest_paths={'soft_mix': manifest},
        raw_dir=RAW, expected_n=QA_COUNT, expected_listed=ELIGIBLE, relation_text_file=RAW / 'rel2text.json')
    row = report['variants']['soft_mix']
    if (row['n_generation_only_excluded'], row['k_min'], row['k_max'], row['lambda_min'], row['lambda_max']) != (
            EXCLUDED, 9, 9, 1.0, 1.0):
        raise ValueError('Soft-Mix must attach nine negatives and a unit weight to every eligible QA')
    hpo.write_json(root / 'wiring.json', {k: v for k, v in row.items() if k not in ('question_ids', 'negatives_by_id')})


def validate_adapters(root, plan):
    summaries, adapters = {}, []
    val_ids = {q['question_id'] for q in hpo.read_json(EVAL)['recurrent_questions'] if q['split'] == 'validation'}
    for arm, variant in ARMS.items():
        train = root / arm / variant
        adapter = train / 'adapter'
        adapters.append(adapter)
        meta = hpo.read_json(adapter / 'run_metadata.json')
        if (meta['trainer_global_step'], meta['trainer_epoch'], meta['qa_samples'], Path(meta['s1_adapter'])) != (
                S2_STEPS, 3.0, QA_COUNT, root / 's1_adapter'):
            raise ValueError(f'{arm} differs from the frozen S2 schedule or starting adapter')
        rows = load_jsonl(train / 'predictions_correct.jsonl')
        if len(rows) != len(val_ids) or {r['question_id'] for r in rows if r['split'] == 'validation'} != val_ids:
            raise ValueError(f'Incomplete validation IDs for {arm}')
        summaries[arm] = wn.full_validation(train / 'summary.json')
    hpo.freeze_text(root / 'selection.json', json.dumps({
        'validation_em': summaries, 'test_adapters': list(ARMS), 'checkpoint_step': S2_STEPS,
        'policy': plan['test_policy'],
    }, indent=2) + '\n')
    return summaries, adapters


def execute(root, plan):
    if not os.environ.get('TMUX'):
        raise RuntimeError('Run this WN pair inside tmux')
    with ExitStack() as stack:
        gpu_lock = stack.enter_context((RUNS / 'ccfb_gpu.lock').open('a'))
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_idle_gpu()
        root.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context((root / 'pipeline.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = root / 'experiment_plan.json'
        if not frozen.exists() and {p.name for p in root.iterdir()} - {'pipeline.lock', 'run.log'}:
            raise ValueError('Refusing an existing run directory without this frozen plan')
        start_available = shutil.disk_usage(root).free + (allocated_bytes(root) if frozen.exists() else 0)
        if start_available < plan['required_free_bytes']:
            raise RuntimeError(f"Run needs {plan['required_free_bytes'] / GIB:.2f} GiB including disk reserve")
        hpo.freeze_text(frozen, json.dumps(plan, sort_keys=True, indent=2) + '\n')
        stage = 'initialization'

        def status(state, **fields):
            hpo.write_json(root / 'run_status.json', {'status': state, 'stage': stage,
                's1_seed': plan['s1_seed'], 's2_seed': plan['s2_seed'],
                'updated_at_utc': datetime.now(timezone.utc).isoformat(), **fields})

        def run(name):
            nonlocal stage
            stage = name
            cmd, outputs = plan['stages'][name]
            receipt = root / 'completed' / (name + '.json')
            if receipt.is_file():
                saved = hpo.read_json(receipt)
                if saved['command'] != cmd or saved['outputs'] != {p: hpo.sha256(p) for p in outputs}:
                    raise ValueError(f'Completed stage changed: {name}')
                print(f'[wn-independent-s1] reuse {name}', flush=True)
                return
            if shutil.disk_usage(root).free < RESERVE:
                raise RuntimeError(f'Less than {RESERVE / GIB:.0f} GiB free before {name}')
            status('RUNNING')
            print(f'[wn-independent-s1] start {name}', flush=True)
            started = time.monotonic()
            with (root / (name + '.log')).open('a') as log:
                subprocess.run(cmd, check=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                               pass_fds=(gpu_lock.fileno(), lock.fileno()))
            hpo.write_json(receipt, {'command': cmd, 'outputs': {p: hpo.sha256(p) for p in outputs},
                                    'wall_seconds': time.monotonic() - started})

        try:
            run('s1')
            s1_sha = hpo.read_json(root / 'completed/s1.json')['outputs'][str(root / 's1_adapter/adapter_model.safetensors')]
            if s1_sha == plan['original_s1_sha256']:
                raise ValueError('The new S1 adapter matches the original weights')
            for name in ('b1', 'score', 'freeze'):
                run(name)
            stage = 'verify_soft_mix'
            verify_soft_mix(root)
            run('soft_mix_1')
            stage = 'verify_validation'
            scores, adapters = validate_adapters(root, plan)
            tests = [root / 'test' / arm for arm in ARMS]
            for arm in ARMS:
                run('test_' + arm)
            stage = 'paired_analysis'
            test_em = [hpo.read_json(test / 'summary.json')['test']['official_em'] for test in tests]
            report = analyze('wn18rr', root, EVAL, *tests, wn.COUNTS['test'], test_em, adapters)
            primary = result_rows(PRIMARY)
            report.update(s1_seed=plan['s1_seed'], s2_seed=2026, s1_sha256=s1_sha, validation_em=scores,
                          delta_em=test_em[1] - test_em[0], original_s1_delta_em=(
                              primary['soft_mix_1']['test_official_em'] - primary['b1_seed2026']['test_official_em']))
            hpo.write_json(root / 'comparison.json', report)
            status('COMPLETE', delta_em=report['delta_em'])
        except (Exception, KeyboardInterrupt) as exc:
            status('INTERRUPTED', error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s1_seed', type=int, default=2027)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.s1_seed < 2**32 or args.s1_seed == 2026:
        parser.error('choose a valid S1 seed other than 2026')
    root = RUNS / f'ccfb_wn_independent_s1_seed{args.s1_seed}_s2seed2026'
    plan = build_plan(root, args.s1_seed)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    execute(root, plan)


if __name__ == '__main__':
    main()

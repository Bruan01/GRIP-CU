#!/usr/bin/env python3
"""Bounded, validation-selected Stage2 warm-start refinement after the main queue.

The prior optimizer was retired. This explicitly starts a NEW optimizer/data
cursor from the preserved step32 adapter; it is not an exact checkpoint resume.
The parent result is immutable and all new results are adaptive exploration.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import supervise_nell_fb_hpo as guard

hpo = guard.hpo
TERMINAL = ('all_three_single_seed_targets_achieved', 'search_budget_finished_targets_not_all_achieved')
RUN_NAME = 'nell23k_stage2_refinement_20261102'


def write(path, value):
    hpo.write_json(path, value)


def progress(out, phase, **extra):
    write(out / 'status.json', {'updated_at': guard.now(), 'pid': os.getpid(),
        'phase': phase, 'search_stage': 2, 'adaptive_after_test_exposure': True, **extra})


def queue_finished(campaign):
    state = guard.read(campaign / 'status.json')
    if state.get('action') not in TERMINAL:
        return False
    for dataset in ('nell23k', 'codexm', 'fb15k237_2'):
        path = campaign / dataset / 'failure.json'
        if path.exists() and not guard.read(path).get('resolved'):
            return False
    return True


def contract(campaign):
    record = guard.read(campaign / 'nell23k/dataset_contract.json')
    parent = Path(record['search_dir'])
    selection = guard.read(parent / 'selection.json')
    winner = selection['winner']
    adapter = Path(winner['adapter'])
    if guard.fingerprint(adapter) != winner['adapter_sha256']:
        raise ValueError('Preserved parent winner identity changed')
    params = {**winner['params'], 'learning_rate': 0.0002}
    assets = [Path(record[key]) for key in ('input', 'valid', 'test', 'relation_eligibility')]
    assets += [Path(record['raw']) / 'train.txt']
    assets += [Path(record['sampler']) / name for name in
               ('counterfactual_bank.jsonl', 'metadata.json', 'wiring.json')]
    return {'schema': 'nell-stage2-warmstart-v1', 'parent_campaign': str(campaign),
        'parent_selection': str(parent / 'selection.json'),
        'parent_full_validation_em': winner['full_validation_em'],
        'parent_screen_em': winner['screen_em'], 'parent_adapter': str(adapter),
        'parent_adapter_sha256': winner['adapter_sha256'],
        'original_stage1_sha256': guard.fingerprint(Path(record['s1_adapter'])),
        'matched_b1_sha256': guard.fingerprint(Path(record['b1_adapter'])),
        'asset_sha256': {str(p): hpo.sha256_file(p) for p in assets},
        'record': record, 'params': params, 'additional_steps': [8, 16, 32],
        'total_stage2_steps': [40, 48, 64], 'seed': 2026,
        'minimum_full_validation_gain': 0.002, 'maximum_full_validation_candidates': 1,
        'optimizer_reset': True, 'data_cursor_reset': True,
        'schedule_horizon_additional_steps': 240,
        'question': 'Does lower-LR Stage2 refinement improve the frozen step32 validation EM?',
        'fixed': 'Task, candidates, generation decoder, seed, rank, loss weight, temperature, weight decay',
        'intervention': 'Warm-start step32; reset optimizer/data cursor; LR=2e-4; select added steps on validation',
        'scope': 'Adaptive exploration after prior test exposure; not an untouched confirmatory test',
        'queue_policy': 'Wait for the existing three-dataset queue, its exclusive lock, and GPU idle',
        'storage_policy': 'Linux volume; top2 validation snapshots; latest resumable state; parent untouched',
        'paper_role': 'reference_only_until_measured', 'claim_update': 'No new score claimed before measurement',
        'stop': 'At most32 added steps; one full-validation finalist; one test only if full-validation gate passes; never reopen'}


def contracts_match(previous, current):
    """Keep frozen semantics/hashes strict while accepting real path aliases."""
    top_paths = ('parent_campaign', 'parent_selection', 'parent_adapter')
    record_paths = ('input', 'valid', 'test', 'raw', 's1_adapter', 'b1_adapter',
                    'baseline_root', 'sampler', 'search_dir', 'relation_eligibility', 'entity_text_file')
    special = (*top_paths, 'record', 'asset_sha256')
    if ({k: v for k, v in previous.items() if k not in special}
            != {k: v for k, v in current.items() if k not in special}):
        return False
    if not all(guard.same_asset_path(previous.get(k), current.get(k)) for k in top_paths):
        return False
    old_record, new_record = previous.get('record', {}), current.get('record', {})
    if ({k: v for k, v in old_record.items() if k not in record_paths}
            != {k: v for k, v in new_record.items() if k not in record_paths}):
        return False
    if not all(guard.same_asset_path(old_record.get(k), new_record.get(k)) for k in record_paths):
        return False
    old_assets, new_assets = previous.get('asset_sha256', {}), current.get('asset_sha256', {})
    if len(old_assets) != len(new_assets):
        return False
    unmatched = dict(new_assets)
    for path, digest in old_assets.items():
        aliases = [other for other, value in unmatched.items()
                   if value == digest and guard.same_asset_path(path, other)]
        if len(aliases) != 1:
            return False
        del unmatched[aliases[0]]
    return not unmatched


def freeze_contract(campaign, out):
    expected = contract(campaign)
    path = out / 'contract.json'
    if path.exists():
        previous = guard.read(path)
        if not contracts_match(previous, expected):
            raise ValueError('Refinement contract or input identities changed')
        return previous
    write(path, expected)
    return expected


def cleanup(out, terminal=False):
    state = SimpleNamespace(number=0, state=SimpleNamespace(name='COMPLETE' if terminal else 'RUNNING'))
    guard.reuse.prune_search_storage(out, [state])


def finish_selected(c, out, selection):
    winner = selection['winner']
    if guard.fingerprint(Path(c['parent_adapter'])) != c['parent_adapter_sha256']:
        raise ValueError('Preserved parent adapter changed during refinement')
    if guard.fingerprint(Path(winner['adapter'])) != winner['adapter_sha256']:
        raise ValueError('Locked refinement adapter changed')
    cleanup(out, terminal=True)
    progress(out, 'selected_full_test', winner=winner)
    em = guard.evaluate(Path(winner['adapter']), Path(c['record']['test']),
                        out / 'selected_test', RUN_NAME + '_selected_test')
    parent = Path(c['record']['search_dir'])
    # These were already computed: no duplicate baseline or parent test inference.
    prior = guard.read(parent / 'final_result.json')
    result = {'dataset': 'nell23k', 'seed': 2026, 'search_stage': 2,
        'selected_test_em': em, 'parent_test_em': prior['selected_test_em'],
        'delta_vs_parent_pp': 100 * (em - prior['selected_test_em']),
        'b1_test_em': prior['b1_test_em'], 'paper_reference_em': c['record']['paper_reference'],
        'delta_vs_paper_pp': 100 * (em - c['record']['paper_reference']),
        'test_count': guard.read(out / 'selected_test/summary.json')['all']['count'],
        'winner': winner, 'selection_file': str(out / 'selection.json'),
        'prediction_file': str(out / 'selected_test/predictions.jsonl'),
        'search_stopped_before_test': True, 'test_used_for_checkpoint_selection': False,
        'adaptive_after_test_exposure': True, 'scope': c['scope'],
        'retain_parent_regardless_of_outcome': True}
    write(out / 'final_result.json', result)
    progress(out, 'completed', result=result)
    return result


def refine(c, out):
    if (out / 'final_result.json').exists():
        return guard.read(out / 'final_result.json')
    if (out / 'no_go.json').exists():
        return guard.read(out / 'no_go.json')
    if (out / 'selection.json').exists():
        return finish_selected(c, out, guard.read(out / 'selection.json'))
    record = c['record']
    paths = {key: Path(record[key]) for key in ('input', 'valid', 'test', 'raw', 'relation_eligibility')}
    paths.update(epochs=record['epochs'], expected_listed=record['expected_listed'],
                 save_steps=16, b1_adapter=Path(record['b1_adapter']))
    parent_adapter, bank = Path(c['parent_adapter']), Path(record['sampler'])
    hpo.validate_assets('nell23k', paths, parent_adapter, bank, guard.PYTHON, hpo.MODEL_CACHE_DEFAULT)
    screen = out / 'valid_screen_1000.json'
    hpo.prepare_validation_screen(paths['valid'], screen, 1000, 2026)
    folder = out / 'trials/trial_0000'
    for step in c['additional_steps']:
        frozen = folder / f'step_{step:03d}/adapter'
        metric = frozen.parent / 'metrics.json'
        if metric.exists():
            continue
        if not guard.adapter_ready(frozen):
            cleanup(out)
            guard.reuse.require_training_space(out)
            resumes = [p for p in (folder / 'train_run/trainer_listed').glob('checkpoint-*')
                       if guard.reuse.complete_checkpoint(p)]
            resume = max(resumes, key=lambda p: int(p.name.split('-')[-1]), default=None)
            if resume and int(resume.name.split('-')[-1]) > step:
                raise RuntimeError('Cannot recover an unmeasured earlier refinement checkpoint')
            progress(out, 'stage2_training', additional_step=step,
                     log=str(folder / f'train_rung_{step}.log'))
            hpo.train_rung(python=guard.PYTHON, dataset='nell23k', paths=paths, adapter=parent_adapter,
                sampler=bank, model_cache=hpo.MODEL_CACHE_DEFAULT, trial_dir=folder,
                trial_number=0, rung_step=step, fresh=resume is None,
                resume_checkpoint=resume, params=c['params'])
            latest = max((p for p in (folder / 'train_run/trainer_listed').glob('checkpoint-*')
                          if guard.reuse.complete_checkpoint(p)), key=lambda p: int(p.name.split('-')[-1]))
            if guard.read(latest / 'trainer_state.json')['global_step'] != step:
                raise RuntimeError('Refinement trainer stopped at an unexpected step')
            partial = frozen.with_name('adapter.copying')
            shutil.copytree(folder / 'train_run/listed/adapter', partial, dirs_exist_ok=True)
            partial.rename(frozen)
        progress(out, 'validation_screen', additional_step=step,
                 log=str(frozen.parent / 'validation_screen/evaluation.log'))
        em = guard.evaluate(frozen, screen, frozen.parent / 'validation_screen', f'{RUN_NAME}_added{step}')
        write(metric, {'trial': 0, 'step': step, 'total_stage2_steps': 32 + step,
            'screen_em': em, 'adapter': str(frozen), 'adapter_sha256': guard.fingerprint(frozen),
            'params': c['params'], 'selected_on': 'validation_screen_1000'})
        cleanup(out)
    candidates = [guard.read(p) for p in folder.glob('step_*/metrics.json')]
    if guard.fingerprint(parent_adapter) != c['parent_adapter_sha256']:
        raise ValueError('Preserved parent adapter changed during refinement')
    winner = max(candidates, key=lambda row: (row['screen_em'], -row['step']))
    if winner['screen_em'] >= c['parent_screen_em']:
        progress(out, 'full_validation', winner=winner,
                 log=str(Path(winner['adapter']).parent / 'full_validation/evaluation.log'))
        winner['full_validation_em'] = guard.evaluate(Path(winner['adapter']), paths['valid'],
            Path(winner['adapter']).parent / 'full_validation', f'{RUN_NAME}_full_validation')
    promoted = winner.get('full_validation_em', -1) >= c['parent_full_validation_em'] + c['minimum_full_validation_gain']
    decision = {'selected_at': guard.now(), 'winner': winner, 'promote': promoted,
        'parent_full_validation_em': c['parent_full_validation_em'], 'selection_split': 'validation_only',
        'search_stopped_before_test': True, 'test_used_for_checkpoint_selection': False,
        'adaptive_after_test_exposure': True, 'scope': c['scope']}
    # This immutable selection is durable BEFORE any test predictions are opened.
    write(out / ('selection.json' if promoted else 'no_go.json'), decision)
    cleanup(out, terminal=True)
    if promoted:
        return finish_selected(c, out, decision)
    progress(out, 'no_validation_improvement', decision=decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=guard.HNG / 'results/hpo/stage2_three_dataset_20261030')
    parser.add_argument('--out', type=Path, default=Path('/home/kieran/grip-cu-runs') / RUN_NAME)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    campaign, out = args.campaign.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a+') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = freeze_contract(campaign, out)
        if args.plan_only:
            print(f'Frozen validation-only selection plan: {out / "contract.json"}')
            return
        if any((out / name).exists() for name in ('final_result.json', 'no_go.json')):
            return
        with (campaign / 'supervisor.lock').open('a+') as queue_lock:
            while True:
                progress(out, 'waiting_for_three_dataset_queue', campaign=str(campaign))
                if queue_finished(campaign):
                    try:
                        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        gpu = guard.probe(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
                        if gpu.get('returncode') != 0:
                            raise RuntimeError(f'Cannot verify GPU availability: {gpu}')
                        if queue_finished(campaign) and not gpu.get('stdout', '').strip():
                            break
                        fcntl.flock(queue_lock, fcntl.LOCK_UN)
                time.sleep(60)
            # Recheck after the potentially long queue delay, before any GPU work.
            c = freeze_contract(campaign, out)
            try:
                refine(c, out)
            except BaseException as error:
                progress(out, 'failed_resume_required', error=str(error))
                raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Confirm existing promising Stage2 checkpoints before expanding HPO."""
from __future__ import annotations

import argparse
from pathlib import Path

import supervise_nell_fb_hpo as guard

MINIMUM_FULL_VALIDATION_GAIN = 0.005


def locked_selection(out):
    path = Path(out) / 'selection.json'
    return path.exists() and guard.read(path).get('search_stopped_before_test') is True


def eligible(record):
    out = Path(record['search_dir'])
    if (out / 'final_result.json').exists():
        return False
    if locked_selection(out):
        return True
    if (out / 'early_validation_decision.json').exists():
        return False
    progress = out / 'search_progress.json'
    if not progress.exists():
        return False
    state = guard.read(progress)
    if state.get('trials_recorded', 0) < 2 or state.get('resource_cap', 0) < 32:
        return False
    baseline = out / 'b1_screen/summary.json'
    metrics = [guard.read(p) for p in (out / 'trials').glob('trial_*/step_*/metrics.json')]
    return (baseline.exists() and metrics and
            max(row['screen_em'] for row in metrics) > guard.read(baseline)['all']['em'])


def live(out, dataset, phase, log, **extra):
    guard.hpo.write_json(out / 'live_progress.json', {
        'dataset': dataset, 'phase': phase, 'search_stage': 2, 'log': str(log), **extra})


def close_search_before_test(out, dataset):
    progress = out / 'search_progress.json'
    state = guard.read(progress) if progress.exists() else {}
    guard.hpo.write_json(progress, {**state, 'search_stopped_before_test': True,
        'selection_mode': 'early_full_validation_gate', 'test_evaluated': False})
    database = out / 'study.sqlite3'
    if not database.exists():
        return
    import optuna
    study = optuna.load_study(study_name=f'{dataset}_sccl_validation_checkpoint_v1',
                              storage=f'sqlite:///{database.as_posix()}')
    for frozen in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        trial = optuna.trial.Trial(study, frozen._trial_id)
        trial.set_user_attr('stop_reason', 'Full-validation early gate passed; search closed before test')
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    guard.hpo.write_study_exports(study, out)
    guard.reuse.prune_search_storage(out, study.get_trials())


def confirm(record):
    dataset = record['dataset']
    out, b1 = Path(record['search_dir']), Path(record['b1_adapter'])
    if (out / 'final_result.json').exists():
        return guard.read(out / 'final_result.json')
    if not eligible(record):
        return None
    expected = guard.read(out / 'fixed_stage1_identity.json')
    if guard.fingerprint(Path(record['s1_adapter'])) != expected:
        raise ValueError('Fixed Stage1 changed before candidate confirmation')
    if locked_selection(out):
        selection = guard.read(out / 'selection.json')
    else:
        metrics = [guard.read(p) for p in (out / 'trials').glob('trial_*/step_*/metrics.json')]
        finalists = sorted(metrics, key=lambda row: (-row['screen_em'], row['trial'], row['step']))[:2]
        for candidate in finalists:
            adapter = Path(candidate['adapter'])
            if not guard.adapter_ready(adapter) or guard.fingerprint(adapter) != candidate['adapter_sha256']:
                raise ValueError('A retained candidate no longer matches its validation checkpoint')
        live(out, dataset, 'early_baseline_full_validation', out / 'b1_full_validation/evaluation.log')
        baseline = guard.evaluate(b1, Path(record['valid']), out / 'b1_full_validation',
                                  f'{dataset}_b1_full_validation')
        for candidate in finalists:
            adapter = Path(candidate['adapter'])
            destination = adapter.parent / 'full_validation'
            live(out, dataset, 'early_candidate_full_validation', destination / 'evaluation.log',
                 trial=candidate['trial'], rung=candidate['step'])
            candidate['full_validation_em'] = guard.evaluate(adapter, Path(record['valid']), destination,
                f"{dataset}_trial{candidate['trial']}_step{candidate['step']}_full_validation")
        winner = max(finalists, key=lambda row: row['full_validation_em'])
        gain = winner['full_validation_em'] - baseline
        promoted = gain + 1e-12 >= MINIMUM_FULL_VALIDATION_GAIN
        selection = {'selected_at': guard.now(), 'selection_split': 'full_validation',
            'finalists': finalists, 'b1_full_validation_em': baseline, 'winner': winner,
            'promote': promoted, 'full_validation_gain_pp': gain * 100,
            'minimum_full_validation_gain_pp': MINIMUM_FULL_VALIDATION_GAIN * 100,
            'test_used_in_this_search_selection': False, 'prior_nell_test_exposure': dataset == 'nell23k',
            'search_stopped_before_test': promoted, 'selection_mode': 'early_full_validation_gate',
            'frozen_adapter_sha256': winner['adapter_sha256'], 'b1_sha256': guard.fingerprint(b1),
            'test_file_sha256': guard.hpo.sha256_file(Path(record['test'])),
            'next_action': 'single_fixed_test_then_stop_this_dataset' if promoted else 'continue_declared_stage2_search'}
        if not promoted:
            guard.hpo.write_json(out / 'early_validation_decision.json', selection)
            # A rejected validation gate must not create a terminal no_go file
            # or open the test split. The remaining declared HPO stays intact.
            return selection
        guard.hpo.write_json(out / 'selection.json', selection)
        guard.hpo.write_json(out / 'early_validation_decision.json', selection)
    winner = selection['winner']
    adapter = Path(winner['adapter'])
    if (guard.fingerprint(adapter) != selection['frozen_adapter_sha256'] or
            guard.fingerprint(b1) != selection['b1_sha256'] or
            guard.hpo.sha256_file(Path(record['test'])) != selection['test_file_sha256']):
        raise ValueError('Locked model/baseline/test identity changed')
    close_search_before_test(out, dataset)
    live(out, dataset, 'early_matched_baseline_test', out / 'b1_test/evaluation.log')
    baseline_test = guard.evaluate(b1, Path(record['test']), out / 'b1_test', f'{dataset}_matched_b1_test')
    live(out, dataset, 'early_selected_full_test', out / 'selected_test/evaluation.log',
         trial=winner['trial'], rung=winner['step'])
    selected_test = guard.evaluate(adapter, Path(record['test']), out / 'selected_test',
                                   f'{dataset}_selected_checkpoint_test')
    reference = record['paper_reference']
    result = {'dataset': dataset, 'claim_type': 'computed', 'winner': winner,
        'b1_test_em': baseline_test, 'selected_test_em': selected_test,
        'paper_reference_em': reference, 'delta_vs_paper_pp': (selected_test - reference) * 100,
        'exceeds_paper_reference': selected_test > reference,
        'improves_matched_baseline': selected_test > baseline_test,
        'target_achieved': selected_test > max(baseline_test, reference),
        'test_count': len(guard.read(Path(record['test']))['questions']),
        'seed': 2026, 'search_stage': 2, 'search_stopped_before_test': True,
        'selection_mode': 'early_full_validation_gate',
        'scope': 'Single seed and fixed local protocol; paper comparability requires protocol audit; test outcome does not reopen this dataset search',
        'prediction_file': str(out / 'selected_test/predictions.jsonl'),
        'selection_file': str(out / 'selection.json'), 'test_used_in_this_search_selection': False}
    guard.hpo.write_json(out / 'final_result.json', result)
    guard.hpo.write_json(out / 'search_progress.json', {
        **guard.read(out / 'search_progress.json'), 'test_evaluated': True,
        'selected_test_em': selected_test, 'target_achieved': result['target_achieved']})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    confirm(guard.read(args.record))

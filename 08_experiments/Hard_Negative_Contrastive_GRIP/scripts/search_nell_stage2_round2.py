#!/usr/bin/env python3
"""Two predeclared Stage2 proposals from the fixed original Stage1, seed2026."""
import argparse
import fcntl
from pathlib import Path
import os
import time

import refine_nell_stage2 as prior
import supervise_nell_fb_hpo as guard

NAME = 'nell23k_stage2_round2_20261104'


def live(out, phase, **extra):
    payload = {'updated_at': guard.now(), 'pid': os.getpid(), 'phase': phase,
               'search_stage': 2, 'adaptive_after_test_exposure': True, **extra}
    guard.hpo.write_json(out / 'status.json', payload)
    guard.hpo.write_json(out / 'live_progress.json', payload)


def make_contract(campaign):
    base = prior.contract(campaign)
    incumbent_file = Path('/home/kieran/grip-cu-runs/nell23k_stage2_refinement_20261102/selection.json')
    incumbent = guard.read(incumbent_file)['winner']
    if guard.fingerprint(Path(incumbent['adapter'])) != incumbent['adapter_sha256']:
        raise ValueError('Validation incumbent identity changed')
    original_selection = guard.read(Path(base['record']['search_dir']) / 'selection.json')
    original = original_selection['winner']
    if base['original_stage1_sha256'] != guard.read(campaign/'nell23k/fixed_stage1.json'):
        raise ValueError('Pinned original Stage1 changed')
    if base['matched_b1_sha256'] != original_selection['b1_sha256']:
        raise ValueError('Pinned original B1 changed')
    params = original['params']
    proposals = [{**params, 'lambda_candidate': .05}, {**params, 'learning_rate': .0006}]
    c = {k: base[k] for k in ('parent_campaign', 'record', 'asset_sha256',
                              'original_stage1_sha256', 'matched_b1_sha256')}
    c.update(schema='nell-stage2-local-search-v2', seed=2026, search_stage=2,
        parent_selection=str(incumbent_file), parent_adapter=incumbent['adapter'],
        parent_adapter_sha256=incumbent['adapter_sha256'],
        initial_trials=proposals, trial_budget=2, rungs=[16, 32, 48], resource_cap=48,
        prune_after=48, maximum_total_optimizer_steps=96, screen_size=1000,
        screen_gate_em=original['screen_em'],
        full_validation_incumbent_em=incumbent['full_validation_em'], minimum_validation_gain=.002,
        full_validation_target_em=incumbent['full_validation_em']+.002,
        maximum_full_validation_candidates=2, maximum_selected_tests=1,
        train_from='fixed original Stage1; fresh Stage2 optimizer, full240-step schedule horizon',
        question='Can lower candidate-loss weight or lower Stage2 learning rate improve validation EM?',
        fixed='Stage1, seed, tasks, candidates, temperature, weight decay, LoRA, decoder, schedule horizon',
        scope='Adaptive exploration after prior NELL test exposure; not an untouched confirmatory test',
        stop='Two proposals, at most48 steps each; full-validation gate; at mostone locked test; never reopen')
    return c


def prepare(campaign, out):
    current = make_contract(campaign)
    path = out / 'contract.json'
    if path.exists():
        frozen = guard.read(path)
        if not prior.contracts_match(frozen, current):
            raise ValueError('Round2 contract/input identities changed')
        return frozen
    guard.hpo.write_json(path, current)
    return current


def finish(c, out, selection):
    winner = selection['winner']
    if guard.fingerprint(Path(winner['adapter'])) != winner['adapter_sha256']:
        raise ValueError('Locked selected weights changed')
    live(out, 'selected_full_test', winner=winner, log=str(out/'selected_test/evaluation.log'))
    em = guard.evaluate(Path(winner['adapter']), Path(c['record']['test']),
                        out/'selected_test', NAME+'_selected_test')
    parent = guard.read(Path(c['record']['search_dir'])/'final_result.json')
    result = {'dataset': 'nell23k', 'seed': 2026, 'search_stage': 2, 'winner': winner,
        'selected_test_em': em, 'b1_test_em': parent['b1_test_em'],
        'paper_reference_em': c['record']['paper_reference'],
        'test_count': guard.read(out/'selected_test/summary.json')['all']['count'],
        'delta_vs_paper_pp': 100*(em-c['record']['paper_reference']),
        'previous_frozen_test_em': parent['selected_test_em'],
        'delta_vs_previous_frozen_run_pp': 100*(em-parent['selected_test_em']),
        'improves_previous_frozen_test': em>parent['selected_test_em'],
        'target_achieved': em>max(parent['b1_test_em'], c['record']['paper_reference']),
        'search_stopped_before_test': True, 'test_used_for_checkpoint_selection': False,
        'scope': c['scope'], 'prediction_file': str(out/'selected_test/predictions.jsonl')}
    guard.hpo.write_json(out/'final_result.json', result)
    live(out, 'completed', result=result)
    return result


def run(campaign, out, c):
    for name in ('final_result.json', 'no_go.json'):
        if (out/name).exists():
            return guard.read(out/name)
    if (out/'selection.json').exists():
        return finish(c, out, guard.read(out/'selection.json'))
    live(out, 'stage2_search', log=str(out/'supervisor.log'))
    spec = {**c['record'], 'search_dir': str(out), 'trial_budget': c['trial_budget'],
            'rungs': c['rungs'], 'resource_cap': c['resource_cap'], 'prune_after': c['prune_after'],
            'initial_trials': c['initial_trials'], 'screen_size': c['screen_size'], 'search_only': True}
    guard.hpo.write_json(out/'worker_spec.json', spec)
    guard.fb_search(campaign, spec)
    candidates = sorted((guard.read(p) for p in (out/'trials').glob('trial_*/step_*/metrics.json')),
                        key=lambda row: (-row['screen_em'], row['trial'], row['step']))[:2]
    if not candidates:
        raise RuntimeError('No measured Stage2 checkpoints')
    finalists = [row for row in candidates if row['screen_em']>=c['screen_gate_em']]
    for row in finalists:
        dest = Path(row['adapter']).parent/'full_validation'
        live(out, 'full_validation', trial=row['trial'], step=row['step'], log=str(dest/'evaluation.log'))
        row['full_validation_em'] = guard.evaluate(Path(row['adapter']), Path(c['record']['valid']),
            dest, f'{NAME}_trial{row["trial"]}_step{row["step"]}_full_validation')
    prepare(campaign, out)
    winner = max(finalists, key=lambda row: row['full_validation_em']) if finalists else candidates[0]
    promote = winner.get('full_validation_em', -1)>=c['full_validation_target_em']
    selected = {'selected_at': guard.now(), 'selection_split': 'validation_only', 'winner': winner,
        'finalists': finalists, 'promote': promote,
        'full_validation_incumbent_em': c['full_validation_incumbent_em'],
        'full_validation_target_em': c['full_validation_target_em'],
        'search_stopped_before_test': True, 'test_used_for_checkpoint_selection': False,
        'frozen_adapter_sha256': winner['adapter_sha256'], 'b1_sha256': c['matched_b1_sha256'],
        'test_file_sha256': c['asset_sha256'][c['record']['test']], 'scope': c['scope']}
    guard.hpo.write_json(out/('selection.json' if promote else 'no_go.json'), selected)
    if promote:
        return finish(c, out, selected)
    live(out, 'no_validation_improvement', decision=selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=guard.HNG/'results/hpo/stage2_three_dataset_20261030')
    parser.add_argument('--out', type=Path, default=Path('/home/kieran/grip-cu-runs')/NAME)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    campaign, out = args.campaign.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        c = prepare(campaign, out)
        if args.plan_only:
            print(out/'contract.json')
            return
        if any((out/name).exists() for name in ('final_result.json','no_go.json')):
            return
        with (campaign/'supervisor.lock').open('a+') as queue_lock:
            while True:
                live(out, 'waiting_for_gpu')
                if prior.queue_finished(campaign):
                    try:
                        fcntl.flock(queue_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        gpu = guard.probe(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
                        if gpu.get('returncode')!=0:
                            raise RuntimeError(f'Cannot inspect GPU: {gpu}')
                        if not gpu.get('stdout','').strip():
                            break
                        fcntl.flock(queue_lock, fcntl.LOCK_UN)
                time.sleep(60)
            try:
                run(campaign, out, prepare(campaign, out))
            except BaseException as error:
                live(out, 'failed_resume_required', error=str(error))
                raise


if __name__=='__main__':
    main()

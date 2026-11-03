#!/usr/bin/env python3
"""Monitor the authorized queue, audit completed tests, and recover NELL workers.

Never kills processes, changes training budgets, reopens a selected search, or
adds proposals in response to observed test scores. Runs in a detached tmux.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import time

import completion_evidence as evidence
import refine_nell_stage2 as refinement
import supervise_nell_fb_hpo as guard
import watch_stage2_queue as main_watch

PAPER = {'nell23k': .8774, 'codexm': .9442, 'fb15k237_2': .8875}
RETAINED = Path('/home/kieran/grip-cu-runs/paper_retained/completion_20261104')


def read(path):
    return guard.read(path) if path.exists() else {}


def worker_alive(state, script, out):
    try:
        proc = Path('/proc') / str(int(state.get('pid', 0)))
        if proc.joinpath('stat').read_text().split()[2] == 'Z':
            return False
        command = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        return (len(command) > 1 and any(Path(p).is_file() and Path(p).samefile(script) for p in command[1:3])
                and '--out' in command and Path(command[command.index('--out') + 1]).samefile(out))
    except (OSError, ValueError, IndexError):
        return False


def recovery_decision(state, alive, occupied, gpu_pids, restarts, maximum=3):
    if state.get('phase') in ('completed', 'no_validation_improvement'):
        return 'finished'
    if alive:
        return 'monitoring'
    if occupied or gpu_pids:
        return 'waiting_for_existing_work'
    if restarts >= maximum:
        return 'retry_limit_reached'
    return 'resume_nell'


def launch_nell(campaign, out):
    session = 'grip_nell_stage2_refine'
    if subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True).returncode == 0:
        return False
    command = 'exec ' + shlex.join([str(guard.PYTHON), '-u',
        str(guard.HNG / 'scripts/refine_nell_stage2.py'), '--campaign', str(campaign), '--out', str(out)])
    command += ' >> ' + shlex.quote(str(out / 'supervisor.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(guard.REPO), command], check=True)
    return True


def existing_report(dataset, result):
    names = {'nell23k': 'nell23k_single_seed_test_20261102.json',
             'codexm': 'codexm_single_seed_test_20261103.json'}
    report = guard.read(guard.REPO / '11_paper/artifacts' / names[dataset])
    evidence.require(report['selected_test_em'] == result['selected_test_em']
        and report['test_count'] == result['test_count']
        and report['exact_match_flags_recomputed'] and report['matched_prompts_targets_ids_verified'],
        'Existing paper audit differs from current result')
    evidence.require(guard.fingerprint(Path(report['retained_adapter'])) == report['retained_weight_sha256']
        == result['winner']['adapter_sha256'], 'Existing retained weights changed')
    return {**report, 'target_achieved': report['selected_test_em'] > max(PAPER[dataset], report['matched_b1_test_em'])}


def inspect(campaign, out, restarts):
    main = read(campaign / 'status.json')
    watchdog = read(campaign / 'watchdog.json')
    nell = read(out / 'status.json')
    report = {'updated_at': guard.now(), 'monitor_pid': os.getpid(), 'phase': 'monitoring',
        'main_action': main.get('action'), 'main_stage': main.get('stage2_trial'),
        'main_watchdog_action': watchdog.get('action'),
        'unresolved_failures': watchdog.get('unresolved_failures', {}),
        'nell_state': nell, 'nell_automatic_restarts': restarts, 'maximum_nell_restarts': 3,
        'training_budget_changed': False, 'active_jobs_killed': False, 'datasets': {}}
    for dataset in PAPER:
        source = campaign / dataset / 'search'
        if not (source / 'final_result.json').exists():
            report['datasets'][dataset] = {'status': 'pending_full_test'}
            continue
        result = guard.read(source / 'final_result.json')
        if dataset in ('nell23k', 'codexm'):
            audit = existing_report(dataset, result)
        else:
            record = guard.read(campaign / dataset / 'dataset_contract.json')
            evidence.require(record['paper_reference'] == PAPER[dataset], 'Paper reference changed')
            selected = guard.read(source / 'selection.json')
            audit = evidence.audit_selected(source, record, source / 'b1_test', RETAINED / dataset,
                                            selected['b1_sha256'], selected['test_file_sha256'])
        report['datasets'][dataset] = {'status': 'audited', 'selected_test_em': audit['selected_test_em'],
            'paper_reference_em': PAPER[dataset], 'delta_vs_paper_pp': audit['delta_vs_paper_pp'],
            'target_achieved': audit['target_achieved'], 'retained_adapter': audit['retained_adapter']}
    if (out / 'final_result.json').exists():
        c = guard.read(out / 'contract.json')
        audit = evidence.audit_selected(out, c['record'], Path(c['record']['search_dir']) / 'b1_test',
            RETAINED / 'nell_refinement', c['matched_b1_sha256'], c['asset_sha256'][c['record']['test']])
        report['nell_refinement'] = {'status': 'audited', 'selected_test_em': audit['selected_test_em'],
                                    'retained_adapter': audit['retained_adapter']}
    elif (out / 'no_go.json').exists():
        report['nell_refinement'] = {'status': 'no_validation_improvement', 'decision': read(out / 'no_go.json')}
    else:
        script = guard.HNG / 'scripts/refine_nell_stage2.py'
        alive = worker_alive(nell, script, out)
        occupied = main_watch.lock_busy(out / 'run.lock')
        pids = [] if alive else main_watch.gpu_processes()
        decision = recovery_decision(nell, alive, occupied, pids, restarts)
        report['nell_recovery_action'] = decision
        if decision == 'retry_limit_reached':
            report.update(phase='needs_attention', error='NELL operational recovery budget exhausted')
        if decision == 'resume_nell':
            if launch_nell(campaign, out):
                report['nell_automatic_restarts'] = restarts + 1
                report['nell_recovery_action'] = 'restarted_nell_worker'
            else:
                report['nell_recovery_action'] = 'waiting_for_existing_session'
    report['all_three_test_targets_achieved'] = all(
        row.get('status') == 'audited' and row.get('target_achieved') for row in report['datasets'].values())
    all_tested = all(row.get('status') == 'audited' for row in report['datasets'].values())
    if not all_tested and not main_watch.controller_alive(main.get('supervisor_pid')) and watchdog.get('action') == 'retry_limit_reached':
        report.update(phase='needs_attention', error='Main queue operational recovery budget exhausted')
    if all_tested and 'nell_refinement' in report:
        report['phase'] = 'completed_single_seed_targets' if report['all_three_test_targets_achieved'] else 'finished_targets_missing'
    # A failed score never reopens the selected search or creates more trials.
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=guard.HNG / 'results/hpo/stage2_three_dataset_20261030')
    parser.add_argument('--out', type=Path, default=Path('/home/kieran/grip-cu-runs') / refinement.RUN_NAME)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    campaign, out = args.campaign.resolve(), args.out.resolve()
    lock = (campaign / 'completion_watchdog.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = campaign / 'completion_watchdog.json'
    restarts = int(read(status_path).get('nell_automatic_restarts', 0))
    while True:
        try:
            report = inspect(campaign, out, restarts)
            restarts = report['nell_automatic_restarts']
        except Exception as error:
            report = {'updated_at': guard.now(), 'monitor_pid': os.getpid(), 'phase': 'needs_attention',
                'error': str(error), 'nell_automatic_restarts': restarts,
                'training_budget_changed': False, 'active_jobs_killed': False}
        guard.hpo.write_json(status_path, report)
        if args.once or report['phase'] in ('completed_single_seed_targets', 'finished_targets_missing'):
            return
        time.sleep(60)


if __name__ == '__main__':
    main()

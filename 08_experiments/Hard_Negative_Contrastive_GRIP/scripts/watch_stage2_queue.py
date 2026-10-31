#!/usr/bin/env python3
"""Watch the background campaign and resume operational failures without duplication."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HNG = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = HNG / 'results/hpo/stage2_three_dataset_20261030'
DATASETS = ('nell23k', 'codexm', 'fb15k237_2')
TERMINAL = ('all_three_single_seed_targets_achieved', 'search_budget_finished_targets_not_all_achieved')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def controller_alive(pid):
    if not pid:
        return False
    try:
        proc = Path(f'/proc/{int(pid)}')
        if proc.joinpath('stat').read_text().split()[2] == 'Z':
            return False
        command = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        return (any(Path(part).name == 'supervise_stage2_three_datasets.py' for part in command)
                and '--worker-spec' not in command)
    except FileNotFoundError:
        return False


def lock_busy(path):
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def gpu_processes():
    probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20)
    if probe.returncode:
        raise RuntimeError(probe.stderr.strip() or 'Cannot inspect GPU')
    return [row.strip() for row in probe.stdout.splitlines() if row.strip()]


def decide(status, alive, locked, gpu_pids, failures, restarts, maximum):
    if alive:
        return 'monitoring'
    if locked or gpu_pids:
        return 'waiting_for_existing_work'
    if status.get('action') in TERMINAL and not failures:
        return 'budget_finished'
    if restarts >= maximum:
        return 'retry_limit_reached'
    return 'resume_queue'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--max-restarts', type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    guard = (root / 'watchdog.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = read(root / 'watchdog.json', {})
    restarts = int(prior.get('automatic_restarts', 0))
    last_restart = float(prior.get('last_restart_epoch', 0))
    while True:
        try:
            status = read(root / 'status.json', {})
            failures = {dataset: value for dataset in DATASETS
                        if (value := read(root / dataset / 'failure.json', {})) and not value.get('resolved')}
            alive = controller_alive(status.get('supervisor_pid'))
            occupied = lock_busy(root / 'supervisor.lock') or lock_busy(root / 'observer.lock')
            pids = [] if alive else gpu_processes()
            action = decide(status, alive, occupied, pids, failures, restarts, args.max_restarts)
            report = {'updated_at': now(), 'watchdog_pid': os.getpid(), 'action': action,
                'controller_pid': status.get('supervisor_pid'), 'controller_alive': alive,
                'campaign_action': status.get('action'), 'existing_gpu_pids': pids,
                'automatic_restarts': restarts, 'maximum_restarts': args.max_restarts,
                'last_restart_epoch': last_restart, 'unresolved_failures': failures,
                'training_budget_changed': False, 'active_jobs_killed': False}
            if action == 'resume_queue' and time.time() - last_restart >= 300:
                result = subprocess.run(['bash', str(HNG / 'configs/run_stage2_three_datasets.sh')],
                                        capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or result.stdout.strip())
                last_restart = time.time()
                if 'Already running:' in result.stdout:
                    report['action'] = 'waiting_for_controller_startup'
                else:
                    restarts += 1
                report.update(automatic_restarts=restarts, last_restart_epoch=last_restart,
                              launcher_output=result.stdout.strip())
                with (root / 'watchdog_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(report, ensure_ascii=False) + '\n')
            write(root / 'watchdog.json', report)
            if action in ('budget_finished', 'retry_limit_reached'):
                return
        except Exception as error:
            write(root / 'watchdog.json', {'updated_at': now(), 'watchdog_pid': os.getpid(),
                'action': 'watchdog_check_error', 'error': str(error),
                'automatic_restarts': restarts, 'last_restart_epoch': last_restart,
                'active_jobs_killed': False})
        time.sleep(max(10, args.interval))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Wait for the existing FB K chain, execute NELL then WN once, and stop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HNG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HNG/'scripts'), str(HNG/'src')]
import run_cross_dataset_controls as driver
import watch_fb_rerank_to_k as previous
from hard_negative_grip.matched_control import read_json, write_json, sha256
from hard_negative_grip.rerank_audit import read_rows, summary, compare

base = driver.base


def status(queue, **fields):
    state = {'at':base.now(), **fields}
    write_json(queue/'status.json', state)
    base.append_event(queue, state)


def validate_plan(plan):
    base.require([s['dataset'] for s in plan['stages']] == ['nell23k','wn18rr']
                 and plan['max_new_training_runs'] == 4 and plan['poll_seconds'] == 30,
                 'unbounded or changed campaign plan')
    base.require(all(s['max_new_training_runs'] == 2 for s in plan['stages']), 'wrong child budgets')


def audit_parent(queue, plan):
    parent = Path(plan['parent_queue'])
    base.require(sha256(parent/'plan.json') == plan['parent_plan_sha256']
                 and read_json(parent/'status.json')['status'] == 'completed', 'FB prerequisite incomplete/changed')
    pp = read_json(parent/'plan.json')
    audits = []
    for stage in pp['stages']:
        audits.append(previous.audit_k(Path(stage['run_dir']), stage['contract_sha256']))
    result = read_json(parent/'paper/result.json')
    base.require(result['status'] == 'completed' and [r['k'] for r in result['metrics']] == [3,9,15], 'FB three-point export incomplete')
    audit = dict(status='pass', at=base.now(), parent_plan_sha256=plan['parent_plan_sha256'],
                 parent_result_sha256=sha256(parent/'paper/result.json'), k_audits=audits)
    write_json(queue/'parent_completion_audit.json', audit)
    return audit


def launch_once(queue, stage):
    run = Path(stage['run_dir'])
    receipt_path = queue/('launch_'+stage['dataset']+'.json')
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        base.require(receipt['status'] == 'launched' and receipt['contract_sha256'] == stage['contract_sha256'],
                     'unresolved/conflicting launch intent; no blind retry')
        return receipt
    base.require(not previous.alive(stage['session']), 'unowned child session exists')
    base.require(sha256(run/'contract.json') == stage['contract_sha256'], 'child contract changed')
    c = base.verify(run)
    driver.prerequisite(c)
    base.require(read_json(run/'status.json')['status'] == 'ready'
                 and read_json(run/'checks/cli.json')['status'] == 'pass', 'child unready/started/failed')
    for arm in driver.ARMS:
        base.require(not list((run/arm/'training/trainer_listed').glob('checkpoint-*'))
                     and not (run/arm/'training/listed/adapter').exists(), 'training already started; do not duplicate')
    argv = [str(base.PYTHON), '-u', str(run/'code_snapshot'/base.REL_HNG/'scripts/run_cross_dataset_controls.py'),
            'run', '--config', str(run/'inputs/design.json'), '--run-dir', str(run)]
    if 'controller_override' in stage:
        override=Path(stage['controller_override'])
        base.require(sha256(override)==stage['controller_override_sha256'],'amended controller changed')
        from source_checkpoint_retention import AMENDMENT
        base.require(sha256(run/AMENDMENT)==stage['retention_amendment_sha256'],'retention amendment changed')
        argv[2]=str(override)
    receipt = dict(status='intent', at=base.now(), contract_sha256=stage['contract_sha256'],
                   session=stage['session'], argv=argv)
    write_json(receipt_path, receipt)
    command = 'exec '+shlex.join(argv)+' >> '+shlex.quote(str(run/'controller.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',stage['session'],'-c',str(run),command], check=True)
    receipt.update(status='launched', launched_at=base.now())
    write_json(receipt_path, receipt)
    return receipt


def audit_child(stage):
    run = Path(stage['run_dir'])
    base.require(sha256(run/'contract.json') == stage['contract_sha256'], 'child contract changed')
    c = base.verify(run)
    result = read_json(run/'paper/result.json')
    saved = read_json(run/'completion_audit.json')
    base.require(saved['status'] == 'pass' and saved['contract_sha256'] == stage['contract_sha256']
                 and saved['archive_sha256'] == sha256(run/'artifact_manifest.json'), 'completion audit identity changed')
    cohort = base.jsonl(run/'inputs/cohort.jsonl')
    points = {}
    for metric in result['metrics']:
        name = metric['method']
        vc = base.verify(run/'ranking'/name)
        identity = read_json(run/'ranking'/name/'models'/name/'identity.json')
        base.require(identity['adapter_sha256'] == vc['models'][name]['adapter_sha256']
                     and identity['cohort_sha256'] == c['cohort_sha256']
                     and identity['code_digest'] == c['code_digest']
                     and identity['candidate_count'] == c['design']['candidate_count'], 'ranking identity changed')
        points[name] = read_rows(run/'ranking'/name/'models'/name/'predictions.jsonl', cohort,
                                 expected_candidates=c['design']['candidate_count'])
        measured = summary(points[name], len(cohort))
        base.require(measured['status'] == 'completed' and all(abs(metric[k+'_percent']-100*measured[k])<1e-10 for k in ('h1','mrr','h3')), 'ranking aggregate changed')
        if name != 'full':
            driver.validate_training(run, c, name)
    for left, right in (('full','random9'),('full','frequency_matched'),('frequency_matched','random9')):
        paired, _ = compare(points[left], points[right], cohort, expected_candidates=c['design']['candidate_count'])
        base.require(paired == result['comparisons'][left+'_minus_'+right]['ranking'], 'paired ranking changed')
    archive = read_json(run/'artifact_manifest.json')
    for entry in archive['artifacts']:
        path = run/entry['path']
        base.require(path.is_file() and path.stat().st_size == entry['bytes'], 'archive missing/size changed')
    return dict(status='pass', at=base.now(), contract_sha256=stage['contract_sha256'],
                archive_sha256=saved['archive_sha256'], metrics=result['metrics'],
                model_question_rows=sum(len(rows) for rows in points.values()),
                artifact_sizes_verified=len(archive['artifacts']))


def check_training_storage(stage, plan):
    if 'training_storage_root' not in stage:
        return
    storage = Path(stage['training_storage_root'])
    for name, target in stage['training_targets'].items():
        logical = Path(stage['run_dir'])/name/'training'
        base.require(logical.is_symlink() and logical.resolve() == Path(target).resolve(),
                     'training output lost its physical-volume redirect')
    base.require(shutil_free(storage) > stage['minimum_training_storage_free_bytes'],
                 'insufficient physical-volume space for all retained dataset checkpoints')
    base.require(shutil_free(Path(plan['host_backing_volume'])) > plan['minimum_host_backing_free_bytes'],
                 'insufficient host backing-volume space for logs/reports')


def cycle(queue, plan):
    """One nonblocking scheduling decision; also exercised by CPU tests."""
    parent = Path(plan['parent_queue'])
    state = read_json(parent/'status.json')
    base.require(state['status'] not in ('failed','blocked'), 'FB K chain failed/blocked; do not skip it')
    if state['status'] != 'completed' or previous.alive(plan['parent_session']):
        base.require(previous.alive(plan['parent_session']) or state['status'] == 'completed', 'FB session vanished; inspect before recovery')
        status(queue, stage='waiting_fb_k', status='running', parent=state)
        return False
    if not (queue/'parent_completion_audit.json').exists():
        audit_parent(queue, plan)
    base.require(read_json(queue/'parent_completion_audit.json')['parent_plan_sha256'] == plan['parent_plan_sha256'], 'cached parent audit mismatch')
    completed = []
    for stage in plan['stages']:
        run = Path(stage['run_dir'])
        child = read_json(run/'status.json')
        audit = queue/(stage['dataset']+'_completion_audit.json')
        if child['status'] == 'completed' and not previous.alive(stage['session']):
            if not audit.exists():
                write_json(audit, audit_child(stage))
            base.require(read_json(audit)['contract_sha256'] == stage['contract_sha256'], 'cached child audit mismatch')
            completed.append(stage['dataset'])
            continue
        base.require(child['status'] not in ('failed','blocked'), 'dataset failed; no automatic retry or later launch')
        receipt = queue/('launch_'+stage['dataset']+'.json')
        if receipt.exists():
            record = read_json(receipt)
            base.require(record['status'] == 'launched' and record['contract_sha256'] == stage['contract_sha256'], 'unresolved launch intent')
            base.require(previous.alive(stage['session']) or child['status'] == 'completed', 'child session vanished before completion')
        elif previous.idle():
            base.require(shutil_free(run) > plan['minimum_free_disk_bytes'], 'insufficient disk for retained checkpoints')
            check_training_storage(stage, plan)
            launch_once(queue, stage)
        status(queue, stage=('running_' if receipt.exists() else 'waiting_gpu_')+stage['dataset'],
               status='running', completed_datasets=completed, child=child)
        return False
    result = dict(status='completed', at=base.now(), new_training_runs=4, datasets=completed,
                  all_signed_results_retained=True, next_action='inspect all outcomes and write; no further automatic experiments')
    write_json(queue/'result.json', result)
    status(queue, stage='completed', **result)
    return True


def shutil_free(path):
    import shutil
    return shutil.disk_usage(path).free


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    queue = a.plan.resolve().parent
    with base.locked(queue):
        plan = read_json(a.plan)
        validate_plan(plan)
        base.require(sha256(a.plan) == (queue/'plan.sha256').read_text().strip(), 'queue plan changed')
        for record in read_json(queue/'code_manifest.json')['files']:
            base.require(sha256(queue/record['path']) == record['sha256'], 'watcher code changed')
        try:
            while not cycle(queue, plan):
                time.sleep(30)
        except BaseException as error:
            status(queue, stage='blocked', status='blocked', error=repr(error), policy='no blind retry or budget expansion')
            raise


if __name__ == '__main__':
    main()

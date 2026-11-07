#!/usr/bin/env python3
"""Run the matched NELL23K B1 with one rolling trainer checkpoint."""

import argparse
import json
from pathlib import Path
import shutil

import run_ccfb_sprint as sprint
from prepare_ccfb_followups import GIB, measure_model_units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry_run', action='store_true')
    options = parser.parse_args()

    root = (sprint.HNG / 'results/runs/ccfb/nell23k/seed2026').resolve()
    args = argparse.Namespace(dataset='nell23k', seed=2026, root=root, shared_root=None)
    plan = sprint.build_plan(args)
    missing = plan['missing_dependencies'] + plan['missing_sources']
    if missing:
        raise ValueError(f'Missing prerequisites: {missing}')
    if plan['arms'] != ['b1'] or [stage[0] for stage in plan['stages']] != ['b1']:
        raise ValueError('Compact checkpoint mode requires the matched B1-only plan')

    stage, command, outputs = plan['stages'][0]
    command = sprint.option(command, '--save_total_limit', 1)
    plan['stages'][0] = (stage, command, outputs)
    plan['training_commands']['b1'] = command
    runner = Path(__file__).resolve()
    plan['source_sha256'][str(runner)] = sprint.hpo.sha256(runner)

    if options.dry_run or not (root / 'experiment_plan.json').is_file():
        model_units, _, _ = measure_model_units()
        sizes = model_units[plan['spec']['lora_r']]
        # A new checkpoint is saved before the previous checkpoint is removed.
        rotation_peak = 2 * sizes['checkpoint']
        export_peak = sizes['checkpoint'] + sizes['final']
        peak = max(rotation_peak, export_peak)
        required = max(12 * GIB, 10 * GIB + peak + (peak + 9) // 10)
        free = shutil.disk_usage(sprint.HNG).free
        if free < required:
            raise RuntimeError(f'Matched B1 needs {required / GIB:.2f} GiB free. '
                               f'{free / GIB:.2f} GiB available')
        if options.dry_run:
            sprint.verify_data(plan)
            sprint.check_idle_gpu()
            print(json.dumps({'plan': plan, 'disk_free_bytes': free,
                              'required_free_bytes': required,
                              'checkpoint_rotation_peak_bytes': rotation_peak,
                              'final_export_peak_bytes': export_peak}, indent=2))
            return

    sprint.execute(args, plan)


if __name__ == '__main__':
    main()

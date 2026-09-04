#!/usr/bin/env python3
"""Select a checkpoint by validation and report its held-out test result."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from priority_distill.io_utils import write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args(); run = a.run_dir.expanduser().resolve()
    rows = [json.loads(x) for x in (run/'controlled_metrics.jsonl').read_text().splitlines() if x.strip()]
    candidates = [r for r in rows if r['checkpoint'] != 'initial']
    if not candidates: raise ValueError('no trained checkpoints found')
    best = max(candidates, key=lambda r: (r['metrics']['graph_free_validation']['accuracy'], -r['epoch']))
    selected = {'selection_rule': 'max graph_free_validation accuracy; test read only from selected checkpoint',
                'selected_checkpoint': best['checkpoint'], 'selected_epoch': best['epoch'],
                'validation': best['metrics']['graph_free_validation'], 'test': best['metrics']['graph_free_test'],
                'oracle_evidence_validation': best['metrics'].get('oracle_evidence_validation'),
                'oracle_evidence_test': best['metrics'].get('oracle_evidence_test'),
                'candidate_selection_validation': best['metrics'].get('candidate_selection_validation'),
                'candidate_selection_test': best['metrics'].get('candidate_selection_test'),
                'candidate_selection_terminal_masked_validation': best['metrics'].get('candidate_selection_terminal_masked_validation'),
                'candidate_selection_terminal_masked_test': best['metrics'].get('candidate_selection_terminal_masked_test')}
    write_json(run/'selected_test_metrics.json', selected)
    summary = json.loads((run/'run_summary.json').read_text())
    lines = [f"# Controlled protocol report", '', f"- protocol: `{summary['protocol']}`", f"- seed: `{summary['seed']}`", f"- selection: validation only", f"- selected checkpoint: `{best['checkpoint']}`", '', '| checkpoint | graph-free validation | graph-free test | oracle validation | oracle test |', '|---|---:|---:|---:|---:|']
    for row in rows:
        m=row['metrics']; lines.append(f"| {row['checkpoint']} | {m['graph_free_validation']['accuracy']:.4f} | {m['graph_free_test']['accuracy']:.4f} | {(m.get('oracle_evidence_validation') or {}).get('accuracy', float('nan')):.4f} | {(m.get('oracle_evidence_test') or {}).get('accuracy', float('nan')):.4f} |")
    lines += ['', '## Selected held-out result', '', f"- graph-free validation: **{best['metrics']['graph_free_validation']['accuracy']:.4f}**", f"- graph-free test: **{best['metrics']['graph_free_test']['accuracy']:.4f}**", '', 'The test value above is read only after selecting the checkpoint by validation.']
    (run/'CONTROLLED_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f"selected={best['checkpoint']} validation={best['metrics']['graph_free_validation']['accuracy']:.4f} test={best['metrics']['graph_free_test']['accuracy']:.4f}")

if __name__ == '__main__': main()

"""Stream frozen B1 scores; reconstruct actual-bank diagnostics without inference."""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HNG = ROOT / '08_experiments/Hard_Negative_Contrastive_GRIP'
HPO = HNG / 'results/hpo/stage2_three_dataset_20261030'
RUNS = Path('/home/kieran/grip-cu-runs')
DATASETS = {'nell23k': 2974, 'codexm': 2715, 'fb15k237_2': 3109, 'wn18rr': 2709}
PROVENANCE = {}
FEATURES = ['structure_raw', 'hardness_raw', 'structure_percentile', 'hardness_percentile']
BINS = np.linspace(0, 1, 11)

def register(path):
    path = Path(path)
    if str(path) not in PROVENANCE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(1024*1024), b''):
                h.update(b)
        PROVENANCE[str(path)] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    return path

def rows(path):
    with register(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def bank(path):
    result = {}
    for r in rows(path):
        q = r['question_id']
        assert q not in result and r['split'] == 'train' and r['seed'] == 2026
        assert len(r['negative_relations']) == len(set(r['negative_relations'])) == 9
        result[q] = r
    return result

def write_csv(name, values):
    with (OUT / 'source_data' / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader(); w.writerows(values)

def main():
    (OUT / 'source_data').mkdir(exist_ok=True)
    hist_rows, grid_rows, summaries, questions, checks = [], [], [], [], {}
    for d, nq in DATASETS.items():
        print('Streaming', d, flush=True)
        asset = (HNG / 'results/runs/wn18rr_sccl_3090_20261023/artifacts/structure_counterfactual_bank'
                 if d == 'wn18rr' else HPO / d / 'assets/k9')
        meta = json.loads(register(asset / 'metadata.json').read_text())
        profiles = json.loads(register(asset / 'relation_profiles.json').read_text())
        rels = sorted(profiles); ri = {r:i for i,r in enumerate(rels)}
        keys = ['edge_count','unique_heads','unique_tails','mean_head_out_degree','mean_tail_in_degree']
        phi = np.array([[math.log1p(profiles[r][k]) for k in keys] for r in rels])
        sim = 1 - np.mean(np.abs(phi[:,None,:]-phi[None,:,:]) /
                          (1+np.abs(phi[:,None,:])+np.abs(phi[None,:,:])), axis=2)
        original = bank(asset / 'counterfactual_bank.jsonl')
        methods = {'full': original}
        audit_pool = None
        if d in ('nell23k','wn18rr'):
            run = RUNS / f'{d}_source_controls_20261107_seed2026'
            ref = bank(run / 'reference/full_bank.jsonl')
            assert set(ref) == set(original)
            assert all(ref[q]['negative_relations'] == original[q]['negative_relations'] for q in ref)
            methods.update(uniform_observed=bank(run / 'bank/random9.jsonl'),
                           quota_observed=bank(run / 'bank/frequency_matched.jsonl'))
            audit_pool = run / 'bank/pool_audit.jsonl'
        elif d == 'fb15k237_2':
            run = RUNS / 'fb_random9_control_20261105_seed2026'
            ref = bank(run / 'reference/counterfactual_bank.jsonl')
            assert set(ref) == set(original)
            assert all(ref[q]['negative_relations'] == original[q]['negative_relations'] for q in ref)
            methods['uniform_observed'] = bank(run / 'bank/random9.jsonl')
            for method, folder, file in [
                ('quota_observed','fb_frequency_matched_control_20261106_seed2026','frequency_matched'),
                ('profile_only_observed','fb_profile_only_control_20261106_seed2026','profile_only'),
                ('violation_only_observed','fb_violation_only_control_20261106_seed2026','violation_only')]:
                methods[method] = bank(RUNS / folder / 'bank' / (file+'.jsonl'))
            audit_pool = run / 'bank/pool_audit.jsonl'
        pools = ({r['question_id']:set(r['legal_relations']) for r in rows(audit_pool)}
                 if audit_pool else None)
        assert len(original) == nq
        for method, b in methods.items():
            assert set(b) == set(original)
            assert all(b[q]['gold_relation'] == original[q]['gold_relation'] for q in original)
        mnames = ['legal_uniform_expected', *methods]
        hist = {m:np.zeros((4,10)) for m in mnames}
        grid = {m:np.zeros((10,10)) for m in mnames}
        totals = {m:np.zeros(6) for m in mnames}
        seen = set(); max_error = 0.; legal_count = 0; unverified = 0
        score_path = Path(meta['score_path'])
        for q, group in itertools.groupby(rows(score_path), key=lambda r:r['qa_id']):
            assert q in original and q not in seen, ('noncontiguous/unknown question',q)
            seen.add(q)
            all_rows = list(group)
            # Historical QA text can omit a namespace; the bank uses the matched label.
            assert all(r['split']=='train' and (r.get('matched_train_relation') or r['gold_relation'])==original[q]['gold_relation'] for r in all_rows), (d,q,'gold label mismatch after mapping')
            valid = [r for r in all_rows if r['is_valid_negative'] and not r['is_gold']]
            names = [r['candidate_relation'] for r in valid]
            assert len(names) == len(set(names)) and len(names)>=9
            if pools is not None:
                assert set(names)==pools[q], ('pool mismatch',d,q)
            legal_count += len(names)
            ix = {r:i for i,r in enumerate(names)}
            s = sim[ri[original[q]['gold_relation']], [ri[r] for r in names]]
            h = np.array([float(r['pairwise_confusion']) for r in valid])
            assert np.isfinite(h).all() and np.all((h>=0)&(h<=1))
            sp = (rankdata(s,method='average')-.5)/len(s)
            hp = (rankdata(h,method='average')-.5)/len(h)
            assert abs(sp.mean()-.5)<1e-12 and abs(hp.mean()-.5)<1e-12
            matrix = np.array([s,h,sp,hp])
            margins = np.array([float(r['candidate_score'])-float(r['gold_score']) for r in valid])
            for item in original[q]['selected_negatives']:
                j = ix[item['relation']]
                err = max(abs(s[j]-item['structure_similarity']),abs(h[j]-item['score_violation']),
                          abs(float(valid[j]['candidate_score'])-item['candidate_score']))
                max_error = max(max_error,err)
                assert err<1e-12, (d,q,item['relation'],err)
            qrow = {'dataset':d,'question_id':q,'pool_size':len(names),
                    'train_pair_verified': methods.get('uniform_observed',{}).get(q,{}).get('train_pair_verified',True)}
            if not qrow['train_pair_verified']:
                unverified += 1
            for m in mnames:
                js = np.arange(len(names)) if m=='legal_uniform_expected' else np.array([ix[r] for r in methods[m][q]['negative_relations']])
                vals = matrix[:,js]
                for i in range(4):
                    hist[m][i] += np.histogram(vals[i],bins=BINS)[0]/len(js)/nq
                grid[m] += np.histogram2d(vals[2],vals[3],bins=(BINS,BINS))[0]/len(js)/nq
                totals[m] += np.r_[vals.mean(axis=1), np.mean(margins[js]>=0), np.mean((vals[2]>=.8)&(vals[3]>=.8))]/nq
                qrow[m+'_hardness_percentile'] = float(vals[3].mean())
                qrow[m+'_structure_percentile'] = float(vals[2].mean())
            questions.append(qrow)
        assert seen == set(original)
        assert unverified == (55 if d=='wn18rr' else 0)
        for m in mnames:
            assert np.allclose(hist[m].sum(axis=1),1,atol=1e-12)
            assert abs(grid[m].sum()-1)<1e-12
            for i,feature in enumerate(FEATURES):
                for b in range(10):
                    hist_rows.append(dict(dataset=d,method=m,feature=feature,bin_left=BINS[b],bin_right=BINS[b+1],probability=hist[m][i,b]))
            for x,y in itertools.product(range(10),repeat=2):
                grid_rows.append(dict(dataset=d,method=m,structure_bin=x,hardness_bin=y,probability=grid[m][x,y]))
            summaries.append(dict(dataset=d,method=m,observed=m!='legal_uniform_expected',questions=nq,
                                  negative_slots=nq*9 if m!='legal_uniform_expected' else '',
                                  legal_candidates=legal_count,**dict(zip(FEATURES,totals[m][:4])),
                                  nonnegative_margin_fraction=totals[m][4],joint_top_quintile_fraction=totals[m][5]))
        checks[d] = dict(questions=nq,full_slots=nq*9,legal_candidates=legal_count,
                         methods=mnames,max_selected_feature_error=max_error,unverified_pairs=unverified,
                         all_bank_choices_legal=True,exact_cached_pool_match=pools is not None,
                         analytic_uniform_mean_percentile=float(totals['legal_uniform_expected'][3]))
        print(d, 'PASS', checks[d], flush=True)
    # Sparse per-question control columns differ across datasets; use a stable union.
    qcols = sorted(set().union(*(r.keys() for r in questions)))
    questions = [{k:r.get(k,'') for k in qcols} for r in questions]
    for name,values in [('histograms.csv',hist_rows),('joint_grid.csv',grid_rows),
                        ('method_summary.csv',summaries),('questions.csv',questions)]:
        write_csv(name,values)
    # Reuse the already independently audited marginal without modifying it.
    src = register(ROOT / '11_paper/artifacts/sampling_distribution_seed2026/source_data/relation_distribution.csv')
    (OUT / 'source_data/relation_distribution.csv').write_bytes(src.read_bytes())
    register(Path(__file__))
    (OUT / 'provenance.json').write_text(json.dumps(PROVENANCE,indent=2)+'\n')
    (OUT / 'validation.json').write_text(json.dumps({'status':'pass','datasets':checks,
        'interpretation':'Frozen B1 generation scores; no final-model calibration or false-negative labels.'},indent=2)+'\n')

if __name__=='__main__':
    main()

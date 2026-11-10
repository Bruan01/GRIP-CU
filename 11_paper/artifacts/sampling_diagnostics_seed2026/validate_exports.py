"""Independent aggregate consistency checks; no rerun of plotting implementation."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def read(name):
    with (HERE/'source_data'/name).open(newline='') as f:
        return list(csv.DictReader(f))

def main():
    summary=read('method_summary.csv'); questions=read('questions.csv')
    hist=defaultdict(dict); joint=defaultdict(lambda:np.zeros((10,10)))
    for r in read('histograms.csv'):
        key=(r['dataset'],r['method'],r['feature'])
        hist[key][round(float(r['bin_left'])*10)]=float(r['probability'])
    for r in read('joint_grid.csv'):
        joint[r['dataset'],r['method']][int(r['structure_bin']),int(r['hardness_bin'])]=float(r['probability'])
    for (d,m),grid in joint.items():
        assert np.all(grid>=0) and abs(grid.sum()-1)<1e-12
        for axis,feature in [(1,'structure_percentile'),(0,'hardness_percentile')]:
            expected=np.array([hist[d,m,feature][i] for i in range(10)])
            assert np.allclose(grid.sum(axis=axis),expected,atol=1e-12)
        for feature in ['structure_raw','hardness_raw','structure_percentile','hardness_percentile']:
            a=np.array([hist[d,m,feature][i] for i in range(10)])
            assert (a>=0).all() and abs(a.sum()-1)<1e-12
    findings={}
    for r in summary:
        d,m=r['dataset'],r['method']
        qs=[q for q in questions if q['dataset']==d]
        assert len(qs)==int(r['questions'])
        for feature in ['structure_percentile','hardness_percentile']:
            a=np.array([float(q[m+'_'+feature]) for q in qs])
            assert np.all((a>=0)&(a<=1))
            assert abs(a.mean()-float(r[feature]))<1e-12
            if m=='legal_uniform_expected': assert np.allclose(a,.5,atol=1e-12)
        if m=='full':
            controls=['uniform_observed','quota_observed'] if d!='codexm' else ['legal_uniform_expected']
            findings[d]={}
            for c in controls:
                delta=np.array([float(q['full_hardness_percentile'])-float(q[c+'_hardness_percentile']) for q in qs])
                findings[d][c]={'mean_difference':float(delta.mean()),'median_difference':float(np.median(delta)),
                    'full_harder_question_fraction':float(np.mean(delta>1e-12)),
                    'equal_question_fraction':float(np.mean(np.abs(delta)<=1e-12))}
    assert len(questions)==11507
    wn=[q for q in questions if q['dataset']=='wn18rr']
    assert sum(q['train_pair_verified']=='False' for q in wn)==55
    result={'status':'pass','questions':len(questions),'method_dataset_groups':len(summary),
            'checks':['histogram unit mass','joint grid unit mass','joint/1D marginal equality',
                      'question/aggregate percentile equality','exact analytic Uniform mean 0.5',
                      'WN historical 55-pair scope'],'paired_hardness':findings,
            'source_data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE/'source_data').glob('*.csv'))}}
    (HERE/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

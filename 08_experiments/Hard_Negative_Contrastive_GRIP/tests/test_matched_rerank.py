"""Synthetic fixtures for rerank identity, ranks, summaries and safe resume."""
from copy import deepcopy
import json,math
from pathlib import Path
import sys
import pytest
HNG=Path(__file__).parents[1];sys.path[:0]=[str(HNG/'src'),str(HNG/'scripts')]
from hard_negative_grip.rerank_audit import SCORE_RULE,rank_fields,audit_row,summary,compare,read_rows
import run_fb_rerank_campaign as queue


def cohort(index=0):
    return dict(dataset_index=index,question_id=f'q{index}',question=f'Q{index}',target=['r2'],candidates=[f'r{i}' for i in range(10)],gold_index=2,frequency_group='low',train_edge_count=10,direct_correct={'full':True,'frequency_matched':False,'random9':False})


def row(index=0,best=2):
    c=cohort(index);raw=[-float(i+1) for i in range(10)];raw[best]=0.;lengths=[3]*10
    return dict(c,split='test',raw_candidate_scores=raw,candidate_token_lengths=lengths,prompt_tokens=100,prompt_token_sha256='a'*64,score_rule=SCORE_RULE,**rank_fields(c['candidates'],raw,2,lengths))


def test_gold_ranks_ties_clipping_and_reciprocal_scores():
    c=cohort();r=rank_fields(c['candidates'],[-100.]*10,2,[3]*10)
    assert r['scores']==[-20.]*10 and r['ranked_indices']==list(range(10))
    assert r['gold_rank']==3 and r['reciprocal_rank']==1/3 and not r['correct'] and r['hits3']
    good=row();assert audit_row(good,c)==good
    s=summary([good,row(1,best=0)],2)
    assert s['status']=='completed' and s['h1']==.5 and s['mrr']<1 and s['h3']==1
    assert summary([good],2)['status']=='partial'


@pytest.mark.parametrize('bad',['nan','inf','length','duplicate','gold'])
def test_invalid_score_inputs_rejected(bad):
    c=cohort();scores=[-1.]*10;lengths=[3]*10;gold=2
    if bad=='nan':scores[1]=math.nan
    if bad=='inf':scores[1]=math.inf
    if bad=='length':lengths[1]=0
    if bad=='duplicate':c['candidates'][1]=c['candidates'][0]
    if bad=='gold':gold=10
    with pytest.raises(ValueError):rank_fields(c['candidates'],scores,gold,lengths)


@pytest.mark.parametrize('bad',['question','target','candidates','rank','flag','score','tokens','rule'])
def test_saved_rows_cannot_relabel_or_change_rank_and_identity(bad):
    r=row();c=cohort()
    if bad=='question':r['question']='different prompt'
    if bad=='target':r['target']=['r1']
    if bad=='candidates':r['candidates']=list(reversed(r['candidates']))
    if bad=='rank':r['gold_rank']=2
    if bad=='flag':r['correct']=1
    if bad=='score':r['scores'][0]=100.
    if bad=='tokens':r['prompt_token_sha256']='bad'
    if bad=='rule':r['score_rule']='sum_log_likelihood'
    with pytest.raises(ValueError):audit_row(r,c)


def test_resume_preserves_prefix_rejects_duplicates_and_repairs_only_terminal_tail(tmp_path):
    p=tmp_path/'predictions.jsonl';coh=[cohort(0),cohort(1)]
    p.write_text(json.dumps(row())+'\n'+json.dumps(row(1))+'\n');assert len(read_rows(p,coh))==2
    p.write_text(json.dumps(row())+'\n'+json.dumps(row())+'\n')
    with pytest.raises(ValueError):read_rows(p,coh,repair_terminal=True)
    p.write_text(json.dumps(row())+'\n{"partial":')
    assert len(read_rows(p,coh,repair_terminal=True))==1
    assert len(list(tmp_path.glob('*.bin')))==1 and p.read_bytes().endswith(b'\n')
    p.write_text('{broken}\n'+json.dumps(row(1))+'\n')
    with pytest.raises(ValueError,match='malformed'):read_rows(p,coh,repair_terminal=True)
    p.write_text(json.dumps(row()));assert len(read_rows(p,coh,repair_terminal=True))==1 and p.read_bytes().endswith(b'\n')


def test_paired_rank_changes_reconcile_and_tokens_must_match():
    coh=[cohort(0),cohort(1)];left=[row(0),row(1,best=0)];right=[row(0,best=0),row(1)]
    result,pairs=compare(left,right,coh)
    assert result['net_correct']==0 and result['paired_counts']['rescue']==result['paired_counts']['regression']==1
    assert abs(result['delta_mrr_pp'])<1e-12
    assert len(pairs)==2 and 'delta_reciprocal_rank' in pairs[0]
    right[0]['prompt_token_sha256']='b'*64
    with pytest.raises(ValueError,match='token/scoring'):compare(left,right,coh)
    with pytest.raises(ValueError,match='incomplete'):compare(left,right[:1],coh)


def test_three_run_order_no_training_and_frozen_worker_entry():
    assert queue.ORDER==['full','frequency_matched','random9']
    cmd=queue.worker_argv(Path('/frozen_run'),'full')
    assert cmd[1]=='-u' and 'code_snapshot' in cmd[2] and 'evaluate_matched_candidate_rerank.py' in cmd[2]
    assert not any('train_listed' in arg for arg in cmd)


def test_completed_three_model_report_and_identity_rejection(tmp_path,monkeypatch):
    run=tmp_path/'synthetic';run.mkdir();coh=[cohort(0),cohort(1)]
    (run/'inputs').mkdir();(run/'inputs/cohort.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in coh))
    c=dict(expected_questions=2,cohort_sha256='cohort',code_digest='code',base_model_identity={'fixture':True},models={m:dict(adapter_sha256={'synthetic':'weights'}) for m in queue.ORDER},repository_root=str(tmp_path/'fake_repo'),experiment_id='SYNTHETIC_ONLY')
    queue.write_json(run/'contract.json',c)
    monkeypatch.setattr(queue.base,'verify',lambda run:c)
    for method in queue.ORDER:
        out=run/'models'/method;out.mkdir(parents=True)
        rows=[row(0),row(1,best=0 if method=='random9' else 2)]
        (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        queue.write_json(out/'summary.json',summary(rows,2))
        queue.write_json(out/'identity.json',dict(adapter_sha256={'synthetic':'weights'},cohort_sha256='cohort',code_digest='code',base_model_identity={'fixture':True},score_rule=SCORE_RULE,score_batch_size=2,max_length=4096,dtype='bfloat16',seed=2026,tie_rule='original_candidate_order'))
    result=queue.report(run,c)
    assert result['training_runs']==0 and len(result['metrics'])==3
    assert result['comparisons']['full_minus_frequency_matched']['delta_h1_pp']==0
    assert result['comparisons']['full_minus_random9']['delta_h1_pp']==50
    assert (run/'paper/table.tex').read_text().count(chr(92)*2)==4
    assert (tmp_path/'fake_repo/11_paper/artifacts/fb_matched_rerank_seed2026/evidence_index.json').exists()
    identity=queue.read_json(run/'models/full/identity.json');identity['adapter_sha256']={'wrong':'weights'};queue.write_json(run/'models/full/identity.json',identity)
    with pytest.raises(ValueError,match='identity mismatch'):queue.report(run,c)


def test_normalized_non_gold_aliases_preserved_but_ambiguous_gold_rejected():
    from hard_negative_grip.rerank_audit import validate_choices
    candidates=['Sports in the Olympic Games','Sports in Olympic Games']+[f'r{i}' for i in range(8)]
    normalize=lambda text:text.lower().replace(' the ',' ')
    gold,collisions=validate_choices(candidates,['r2'],normalize)
    assert gold==4 and len(collisions)==1 and collisions[0]['indices']==[0,1]
    assert len(candidates)==10
    with pytest.raises(ValueError,match='gold must match exactly one'):validate_choices(candidates,[candidates[0]],normalize)

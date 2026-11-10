"""Synthetic checks for weighted selection and the real training metadata path."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

HNG=Path(__file__).parents[1]
sys.path[:0]=[str(HNG/'src'),str(HNG/'scripts')]
from hard_negative_grip.frequency_only import frequency_only_bank_row, validate_frequency_only_manifest
from hard_negative_grip.task_file import build_qa_assets_from_task_texts
from hard_negative_grip.matched_control import MATCHED_FIELDS
import run_fb_frequency_only_control as controller


def options():
    vocabulary=['gold','known']+[f'r{i}' for i in range(15)]
    return dict(question_id='task_qa:0',positive_relation='gold',matched_train_relation='gold',
                relation_order=vocabulary,known_relations={'gold','known'},
                frequencies={n:2**(i%7) for i,n in enumerate(vocabulary)},seed=2026,k=9)


def test_deterministic_order_invariant_legal_unique_with_exact_remaining_probabilities():
    opts=options();row,pool=frequency_only_bank_row(**opts)
    other=deepcopy(opts);other['relation_order'].reverse()
    assert frequency_only_bank_row(**other)==(row,pool)
    assert len(set(row['negative_relations']))==9
    assert not {'gold','known'} & set(row['negative_relations'])
    total=sum(opts['frequencies'][n] for n in pool['legal_relations'])
    assert row['pool_total_frequency']==total
    for d in row['selected_negatives']:
        assert d['conditional_probability']==opts['frequencies'][d['relation']]/total
        total-=opts['frequencies'][d['relation']]
    # The API has no Full bank, structural feature or score argument.
    with pytest.raises(TypeError):frequency_only_bank_row(**opts,parent_row={})
    with pytest.raises(TypeError):frequency_only_bank_row(**opts,candidate_score=100)


def test_one_draw_observed_distribution_tracks_train_frequency_not_relation_uniform():
    opts=dict(positive_relation='gold',matched_train_relation='gold',relation_order=['gold','a','b'],
              known_relations={'gold'},frequencies={'gold':1000000,'a':1,'b':9},seed=2026,k=1)
    heavy=sum(frequency_only_bank_row(question_id=f'task_qa:{i}',**opts)[0]['negative_relations']==['b'] for i in range(4000))
    assert .87 < heavy/4000 < .93


@pytest.mark.parametrize('bad',['count','conditional','initial','total','source','split','parent','uses_full','duplicate','weight'])
def test_rejects_tampered_provenance_and_probabilities(bad):
    row,_=frequency_only_bank_row(**options())
    if bad=='count':row['selected_negatives'][0]['train_edge_count']=0
    if bad=='conditional':row['selected_negatives'][1]['conditional_probability']=.99
    if bad=='initial':row['selected_negatives'][1]['initial_probability']=.99
    if bad=='total':row['pool_total_frequency']+=1
    if bad=='source':row['negative_source']='uniform'
    if bad=='split':row['filter_splits']=['train','test']
    if bad=='parent':row['parent_slot_bins']=[1]*9
    if bad=='uses_full':row['uses_full_bank']=True
    if bad=='duplicate':row['negative_relations'][1]=row['negative_relations'][0]
    if bad=='weight':row['lambda_candidate']=3
    with pytest.raises(ValueError):validate_frequency_only_manifest({row['question_id']:row})


def test_rejects_incomplete_counts_and_legal_pool_shortfall():
    opts=options();opts['frequencies'].pop('r0')
    with pytest.raises(ValueError):frequency_only_bank_row(**opts)
    opts=options();opts['known_relations']=set(opts['relation_order'])-{'r0'}
    with pytest.raises(ValueError,match='shortfall'):frequency_only_bank_row(**opts)


def test_actual_task_builder_preserves_text_gold_eligibility_and_global_loss_weight():
    opts=options();row,_=frequency_only_bank_row(**opts)
    text='<|im_start|>user\nwhat is the relation between a and b?<|im_end|>\n<|im_start|>assistant\n<answer>gold</answer><|im_end|>'
    out,metas=build_qa_assets_from_task_texts([text],seed=2026,relation_order=opts['relation_order'],
                 listed_negative_source='frequency_only',random_manifest={'task_qa:0':row},eligible_question_ids={'task_qa:0'})
    assert out==[text] and metas[0]['listed_relations']==row['negative_relations']
    assert metas[0]['lambda_candidate'] is None and metas[0]['positive_relation']=='gold'
    with pytest.raises(ValueError,match='exactly'):
        build_qa_assets_from_task_texts([text],seed=2026,relation_order=opts['relation_order'],
                 listed_negative_source='frequency_only',random_manifest={'task_qa:0':row},eligible_question_ids=set())


def test_command_retains_training_budget_and_four_checkpoints():
    ref={k:0 for k in MATCHED_FIELDS}
    ref.update(seed=2026,target_modules=['up_proj'],involve_qa_epochs=5.,stop_after_step=32,listed_negative_k=9)
    cmd=controller.make_training_argv(ref,Path('/synthetic/run'),'cache')
    for flag,value in [('--listed_negative_source','frequency_only'),('--negative_sampling_mode','random'),
                       ('--stop_after_step','32'),('--involve_qa_epochs','5.0'),('--save_total_limit','4')]:
        assert cmd[cmd.index(flag)+1]==value
    assert '--max_steps' not in cmd and '--defer_eval' in cmd


def test_report_keeps_direct_and_ranking_directions_and_all_reference_arms(tmp_path,monkeypatch):
    import json
    from hard_negative_grip.rerank_audit import rank_fields,SCORE_RULE
    run=tmp_path/'synthetic';run.mkdir()
    save=lambda p,v:controller.write_json(run/p,v)
    dump=lambda p,rows:(run/p).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(controller,'validate_training',lambda *a:None)
    monkeypatch.setattr(controller.base,'verify',lambda *a:None)
    monkeypatch.setattr(controller,'DESTINATION',tmp_path/'export')
    save('inputs/test.json',dict(id='synthetic',split='test',graph={},questions=[f'q{i}' for i in range(10000)],answers=['gold']*10000))
    choices=['gold']+[f'r{i}' for i in range(9)]
    cohort=[dict(dataset_index=i,question_id=f'synthetic:test:{i}',question=f'q{i}',target=['gold'],candidates=choices,
                 gold_index=0,frequency_group='low',train_edge_count=1,
                 direct_correct=dict(full=i<9000,frequency_matched=i<8990,random9=i<8600)) for i in range(10000)]
    (run/'reference').mkdir();dump('reference/rerank_cohort.jsonl',cohort)
    normalize=run/'code_snapshot/13_base_method/grip-exp/evaluation/utils.py'
    normalize.parent.mkdir(parents=True);normalize.write_text('def normalize_answer(x): return x.lower()\n')
    weights={'synthetic':'fixture'}
    save('selection.json',dict(adapter_sha256=weights))
    save('evaluation/test/reuse_identity.json',dict(adapter=weights))
    dump('evaluation/test/predictions.jsonl',[dict(question_id=r['question_id'],question=r['question'],target=r['target'],split='test',response='gold' if i<8800 else 'wrong',correct=i<8800) for i,r in enumerate(cohort)])
    def ranks(correct):
        out=[]
        for i,r in enumerate(cohort):
            raw=[-1.,-2.]+[-3.]*8 if i<correct else [-2.,-1.]+[-3.]*8
            out.append(dict(r,split='test',raw_candidate_scores=raw,candidate_token_lengths=[1]*10,
                       prompt_tokens=2,prompt_token_sha256='a'*64,score_rule=SCORE_RULE,
                       **rank_fields(choices,raw,0,[1]*10)))
        return out
    for method,n in [('full',9500),('frequency_matched',9490),('random9',9300)]:
        p=run/f'reference/ranking/{method}';p.mkdir(parents=True)
        dump(f'reference/ranking/{method}/predictions.jsonl',ranks(n))
    save('rerank/models/frequency_only/identity.json',dict(adapter_sha256=weights))
    dump('rerank/models/frequency_only/predictions.jsonl',ranks(9600))
    for name in ('preparation_audit.json',):save('bank/'+name,dict(synthetic=True))
    for name in ('relation_frequencies.csv','selected_candidates.csv'):(run/'bank'/name).write_text('synthetic\n')
    save('contract.json',dict(synthetic=True))
    c=dict(experiment_id='SYNTHETIC_ONLY')
    result=controller.report(run,c)
    assert result['comparisons']['full_minus_frequency_only']['direct']['delta_left_minus_right_pp']==2.0
    assert result['comparisons']['full_minus_frequency_only']['ranking']['delta_h1_pp']==-1.0
    assert len(result['metrics'])==4 and result['training_runs']==1
    assert isinstance(controller.read_json(run/'artifact_manifest.json')['artifacts'],list)
    save('evaluation/test/reuse_identity.json',dict(adapter={'changed':'fixture'}))
    with pytest.raises(ValueError,match='identity mismatch'):controller.report(run,c)

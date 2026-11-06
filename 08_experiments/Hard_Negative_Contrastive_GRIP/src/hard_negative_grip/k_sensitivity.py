"""Guards for a coupled K=3/9/15 Full SC-CL sensitivity study."""
from __future__ import annotations
from .matched_control import parameter_diff
from .counterfactuals import sample_structure_counterfactual_group
from .frozen_random import pool_digest

FULL_SAMPLER=dict(uniform_mix=1/3,structure_weight=1.,violation_weight=1.,temperature=.25)


def matched_k_parameters(reference,actual,k):
    if k not in (3,15) or reference['listed_negative_k']!=9:raise ValueError('only K3/K15 replacements of Full K9 are registered')
    expected=dict(reference,listed_negative_k=k)
    rows=parameter_diff(expected,actual)
    return [dict(parameter=r['parameter'],full_k9_reference=reference[r['parameter']],expected=r['reference'],actual=r['random_control'],matches_expected=r['equal'],intended_change=r['parameter']=='listed_negative_k') for r in rows]


def build_k_group(group,profiles,vocabulary,parent,pool,k):
    if k not in (3,15):raise ValueError('unsupported K')
    qid=parent['question_id'];legal=pool['legal_relations'];gold=parent['matched_train_relation'];known=set(pool['excluded_train_relations'])
    if (pool['question_id']!=qid or pool_digest(legal)!=pool['pool_sha256']
            or not group or any(r.get('qa_id')!=qid or r.get('split')!='train' for r in group)
            or len({r['candidate_relation'] for r in group})!=len(group)
            or {r['candidate_relation'] for r in group}!=set(vocabulary)
            or {r['candidate_relation'] for r in group if r.get('is_valid_negative') is True}!=set(legal)
            or set(legal)!=set(vocabulary)-known-{gold} or len(legal)<k):
        raise ValueError('K sensitivity score/pool identity or capacity mismatch')
    for r in group:
        if (r.get('matched_train_relation')!=gold or r.get('gold_relation')!=parent['positive_relation']
                or r.get('is_gold') is not (r['candidate_relation']==gold)):
            raise ValueError('gold mismatch')
    replay,audit9=sample_structure_counterfactual_group(group,profiles,total_k=9,seed=2026,known_relations=known,**FULL_SAMPLER)
    if replay['negative_relations']!=parent['negative_relations']:raise ValueError('Full K9 exact replay failed')
    row,audit=sample_structure_counterfactual_group(group,profiles,total_k=k,seed=2026,known_relations=known,**FULL_SAMPLER)
    a,b=(row['negative_relations'],parent['negative_relations']) if k==3 else (parent['negative_relations'],row['negative_relations'])
    if a!=b[:len(a)]:raise ValueError('nested seeded draw prefix changed')
    return row,audit,dict(question_id=qid,k=k,pool_size=len(legal),full_k9_replayed=True,nested_prefix_verified=True,
                          k_structure_mean=audit['selected_mean_structure_similarity'],k9_structure_mean=audit9['selected_mean_structure_similarity'],
                          k_violation_mean=audit['selected_mean_score_violation'],k9_violation_mean=audit9['selected_mean_score_violation'])

"""Strict CPU audits for matched, fixed-list candidate reranking."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path

SCORE_RULE='mean_log_probability_of_tag_wrapped_answer_tokens_then_clip_20'


def prompt_digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def validate_choices(candidates,targets,normalize,*,expected_candidates=10):
    if expected_candidates<2 or len(candidates)!=expected_candidates or len(set(candidates))!=expected_candidates:
        raise ValueError('require the declared number of distinct original candidate strings')
    names=[normalize(n) for n in candidates];gold=[i for i,n in enumerate(names) if n in {normalize(t) for t in targets}]
    if len(gold)!=1:raise ValueError('gold must match exactly one original candidate under EM normalization')
    collisions=[dict(normalized=name,indices=[i for i,n in enumerate(names) if n==name],original_candidates=[candidates[i] for i,n in enumerate(names) if n==name]) for name in dict.fromkeys(names) if names.count(name)>1]
    return gold[0],collisions


def rank_fields(candidates,raw_scores,gold_index,token_lengths,*,expected_candidates=10):
    n=expected_candidates
    if n<2 or len(candidates)!=n or len(set(candidates))!=n or len(raw_scores)!=n or len(token_lengths)!=n:
        raise ValueError('require the declared candidate, score and length counts')
    if type(gold_index) is not int or not 0<=gold_index<n:raise ValueError('invalid gold index')
    if any(not math.isfinite(v) for v in raw_scores) or any(type(n) is not int or n<1 for n in token_lengths):
        raise ValueError('nonfinite score or invalid answer-token length')
    clipped=[min(20.,max(-20.,float(v))) for v in raw_scores]
    order=sorted(range(n),key=lambda i:(-clipped[i],i))
    rank=order.index(gold_index)+1;best=order[0]
    return dict(scores=clipped,ranked_indices=order,response=candidates[best],correct=rank==1,gold_rank=rank,
                reciprocal_rank=1/rank,hits3=rank<=3,hits10=rank<=10,
                score_margin=clipped[best]-clipped[order[1]],gold_minus_best_negative=clipped[gold_index]-max(v for i,v in enumerate(clipped) if i!=gold_index))


def audit_row(row,cohort,*,expected_candidates=10):
    for key in ('dataset_index','question_id','question','target','candidates','gold_index','frequency_group','train_edge_count','direct_correct'):
        if row.get(key)!=cohort[key]:raise ValueError('rerank cohort mismatch: '+key)
    if row.get('score_rule')!=SCORE_RULE or row.get('split')!='test':raise ValueError('score rule/split mismatch')
    computed=rank_fields(row['candidates'],row['raw_candidate_scores'],row['gold_index'],row['candidate_token_lengths'],expected_candidates=expected_candidates)
    for key,value in computed.items():
        if row.get(key)!=value:raise ValueError('rerank score/rank/flag mismatch: '+key)
    if (type(row.get('correct')) is not bool or type(row.get('hits3')) is not bool
            or not isinstance(row.get('prompt_token_sha256'),str) or len(row['prompt_token_sha256'])!=64
            or type(row.get('prompt_tokens')) is not int or row['prompt_tokens']<1):
        raise ValueError('invalid token identity or boolean flag')
    return row


def read_rows(path,cohort,*,repair_terminal=False,expected_candidates=10):
    """Only a malformed final non-newline tail may be repaired, with a backup."""
    if not path.exists():return []
    raw=path.read_bytes();lines=raw.splitlines(keepends=True);rows=[]
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            if not (repair_terminal and i==len(lines)-1 and not line.endswith(b'\n')):raise ValueError('malformed prediction JSONL')
            backup=path.with_name(path.name+'.terminal-tail-'+hashlib.sha256(raw).hexdigest()[:12]+'.bin')
            backup.write_bytes(raw)
            # Resume repairs only the trailing partial row, never drops bad middle rows.
            path.write_bytes(b''.join(lines[:-1]));break
        if len(rows)>=len(cohort):raise ValueError('unexpected extra prediction row')
        audit_row(row,cohort[len(rows)],expected_candidates=expected_candidates)
        rows.append(row)
    if repair_terminal and rows and raw and not raw.endswith(b'\n') and len(rows)==len(lines):
        path.write_bytes(raw+b'\n')
    return rows


def summary(rows,expected):
    n=len(rows)
    return dict(status='completed' if n==expected else 'partial',n=n,expected=expected,
                h1=sum(r['correct'] for r in rows)/n if n else None,
                mrr=sum(1/r['gold_rank'] for r in rows)/n if n else None,
                h3=sum(r['gold_rank']<=3 for r in rows)/n if n else None,
                h10=sum(r['gold_rank']<=10 for r in rows)/n if n else None)


def compare(left,right,cohort,*,expected_candidates=10):
    if len(left)!=len(right) or len(left)!=len(cohort):raise ValueError('incomplete paired cohorts')
    counts=dict(both_correct=0,rescue=0,regression=0,both_wrong=0);pairs=[]
    for l,r,c in zip(left,right,cohort):
        audit_row(l,c,expected_candidates=expected_candidates);audit_row(r,c,expected_candidates=expected_candidates)
        for key in ('prompt_token_sha256','prompt_tokens','candidate_token_lengths','score_rule'):
            if l[key]!=r[key]:raise ValueError('paired token/scoring identity mismatch: '+key)
        outcome='both_correct' if l['correct'] and r['correct'] else 'rescue' if l['correct'] else 'regression' if r['correct'] else 'both_wrong'
        counts[outcome]+=1
        pairs.append(dict(question_id=c['question_id'],left_rank=l['gold_rank'],right_rank=r['gold_rank'],left_correct=int(l['correct']),right_correct=int(r['correct']),outcome=outcome,delta_reciprocal_rank=1/l['gold_rank']-1/r['gold_rank']))
    n=len(left)
    return dict(n=n,paired_counts=counts,net_correct=counts['rescue']-counts['regression'],delta_h1_pp=100*(counts['rescue']-counts['regression'])/n,delta_mrr_pp=100*sum(r['delta_reciprocal_rank'] for r in pairs)/n),pairs

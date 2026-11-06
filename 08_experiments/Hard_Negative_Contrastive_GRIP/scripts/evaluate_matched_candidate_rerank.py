#!/usr/bin/env python3
"""Inference-only matched reranking with strict identity and per-question scores."""
from __future__ import annotations
import argparse,json,math,os
from pathlib import Path
import sys,time

HNG=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HNG/'scripts'),str(HNG/'src')]
import run_fb_random9_control as base
from hard_negative_grip.matched_control import read_json,write_json,sha256
from hard_negative_grip.rerank_audit import SCORE_RULE,prompt_digest,rank_fields,audit_row,read_rows,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--model',required=True);a=p.parse_args()
    run=a.run_dir;c=base.verify(run);model_spec=c['models'][a.model];out=run/'models'/a.model
    with base.locked(out):
        cohort=base.jsonl(run/'inputs/cohort.jsonl')
        weights={name:sha256(Path(model_spec['adapter_dir'])/name) for name in ('adapter_config.json','adapter_model.safetensors')}
        base.require(weights==model_spec['adapter_sha256'],'source adapter changed')
        identity=dict(model=a.model,adapter_sha256=weights,cohort_sha256=c['cohort_sha256'],code_digest=c['code_digest'],score_rule=SCORE_RULE,
                      score_batch_size=c['score_batch_size'],max_length=c['max_length'],seed=2026,dtype='bfloat16',tie_rule='original_candidate_order',base_model_identity=c['base_model_identity'])
        if (out/'identity.json').exists():base.require(read_json(out/'identity.json')==identity,'resume model/source/scorer identity changed')
        else:base.require(not (out/'predictions.jsonl').exists(),'predictions without identity');write_json(out/'identity.json',identity)
        rows=read_rows(out/'predictions.jsonl',cohort,repair_terminal=True)
        if len(rows)==len(cohort):
            write_json(out/'summary.json',summary(rows,len(cohort)));print('Already complete; no repeated inference.',flush=True);return
        # These imports use the exact frozen evaluation loader and scoring implementation.
        import torch
        from train_listed_contrastive import load_adapter,GRIPEvalDataset
        from hard_negative_grip.task_file import load_graph_record
        from grip.training.candidate_scorer import score_candidate_prompt
        from constants import ANSWER_TAG
        from utils import set_random_seed
        from evaluation.utils import normalize_answer
        set_random_seed(2026)
        args=argparse.Namespace(model_name='qwen-7b',model_cache_dir=c['model_cache'],dtype='bfloat16')
        model=tokenizer=None;started=time.monotonic();initial=len(rows)
        try:
            model,tokenizer=load_adapter(args,Path(model_spec['adapter_dir']),trainable=False)
            base.require(torch.cuda.is_available() and next(model.parameters()).device.type=='cuda','GPU evaluation required; no CPU offload')
            model.requires_grad_(False)
            base.require(all(not param.requires_grad for param in model.parameters()),'inference model unexpectedly trainable')
            model.eval();model.config.use_cache=False
            record=load_graph_record(run/'inputs/test.json')
            dataset=GRIPEvalDataset(questions=record['questions'],answers=record['answers'],tokenizer=tokenizer,graph=record['graph'],title=record.get('title','nell23k'),no_graph_context=True)
            torch.cuda.reset_peak_memory_stats();scoring_started=time.monotonic()
            with (out/'predictions.jsonl').open('a',encoding='utf-8') as stream,torch.inference_mode():
                for i in range(len(rows),len(cohort)):
                    item=cohort[i];input_ids,question,target=dataset[i]
                    targets=[str(v) for v in target] if isinstance(target,list) else [str(target)]
                    base.require(question==item['question'] and targets==item['target'],'tokenized dataset prompt/target mismatch')
                    ids=input_ids[0].tolist();token_count=len(ids)
                    expected_lengths=[len(tokenizer(f'<{ANSWER_TAG}>{name}</{ANSWER_TAG}>',add_special_tokens=False)['input_ids']) for name in item['candidates']]
                    base.require(token_count+max(expected_lengths)<=c['max_length'],'unexpected long prompt; no silent truncation or row exclusion')
                    result=score_candidate_prompt(model,tokenizer,input_ids.to(next(model.parameters()).device),item['candidates'],score_clip=None,candidate_batch_size=c['score_batch_size'])
                    raw=[float(v) for v in result.scores[0].detach().cpu().tolist()];lengths=[int(v) for v in result.token_lengths[0].detach().cpu().tolist()]
                    base.require(lengths==expected_lengths,'scorer answer-token accounting changed')
                    fields=rank_fields(item['candidates'],raw,item['gold_index'],lengths)
                    row=dict(item,split='test',raw_candidate_scores=raw,candidate_token_lengths=lengths,prompt_tokens=token_count,prompt_token_sha256=prompt_digest(ids),score_rule=SCORE_RULE,**fields)
                    base.require(any(normalize_answer(row['response'])==normalize_answer(t) for t in targets)==row['correct'],'rank/EM disagreement')
                    audit_row(row,item)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno());rows.append(row)
                    elapsed=time.monotonic()-scoring_started;done=len(rows)-initial
                    heartbeat=dict(status='running',completed=len(rows),expected=len(cohort),scoring_elapsed_seconds=round(elapsed,3),questions_per_second=done/elapsed,
                                   remaining_seconds_this_model=(len(cohort)-len(rows))*elapsed/done,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,at=base.now())
                    write_json(out/'heartbeat.json',heartbeat)
                    if len(rows)==32 and not (out/'pilot.json').exists():
                        write_json(out/'pilot.json',dict(heartbeat,pilot_questions=32,pilot_reused_in_full_result=True,quality_not_used_for_run_selection=True))
                    if len(rows)==1 or len(rows)%100==0:print(json.dumps(heartbeat),flush=True)
            result=summary(rows,len(cohort));result.update(at=base.now(),model=a.model,total_seconds=round(time.monotonic()-started,3),new_questions=len(rows)-initial)
            write_json(out/'summary.json',result);write_json(out/'heartbeat.json',dict(status='completed',completed=len(rows),expected=len(cohort),at=base.now()))
            print(json.dumps(result,indent=2),flush=True)
        finally:
            if model is not None:del model
            if tokenizer is not None:del tokenizer
            if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':main()

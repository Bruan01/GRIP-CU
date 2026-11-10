"""Build manuscript tables from committed, audited experiment evidence (no training)."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
PAPER=ROOT/'11_paper';DEST=Path(__file__).resolve().parent
END=chr(92)*2

def read(name):return json.loads((PAPER/'artifacts'/name).read_text())
def emit(filename,caption,label,columns,header,rows):
    text=['\\begin{table}[htbp]','\\caption{'+caption+'}','\\label{'+label+'}','\\centering\\small','\\begin{tabular}{'+columns+'}','\\toprule',header+' '+END,'\\midrule']
    text+=[' & '.join(map(str,row))+' '+END for row in rows]
    text+=['\\bottomrule','\\end{tabular}','\\end{table}']
    (DEST/filename).write_text('\n'.join(text)+'\n')

source_names=['nell23k_single_seed_test_20261102.json','codexm_single_seed_test_20261103.json','fb15k237_2_single_seed_test_20261104.json']
rows=[]
for label,name in zip(['NELL23K','CoDEx-M','FB15K237-2'],source_names):
    r=read(name);n=r['test_count'];a=r['selected_correct'];b=r['baseline_correct'];assert a==r['paired_counts']['both_correct']+r['paired_counts']['rescue']
    rows.append([label,f'{n:,}',f'{100*b/n:.2f}',f'{100*a/n:.2f}',f'{100*(a-b)/n:+.2f}',f'{a-b:+d}'])
emit('latest_direct.tex','三数据集本地协议的完整测试直接解码结果（\\%，seed=2026）。B1与方法的问题、选项顺序和目标对齐；方法为验证选定的Stage~2配置。$\\Delta$为方法减B1的百分点差。训练调参及部分优化设置同时变化，故该表不隔离候选损失或采样机制。','tab:latest-direct','lrrrrr','数据集 & $n$ & B1 EM & SC-CL EM & $\\Delta$ (pp) & 净增题数',rows)
r=read('fb_frequency_matched_control_seed2026/result.json');comp=read('fb_component_ablation_seed2026/result.json');rand=read('fb_random9_control_seed2026/result.json');freqonly=read('fb_frequency_only_control_seed2026/result.json')
assert freqonly['status']=='completed' and freqonly['n']==10000 and freqonly['seed']==2026
methods=comp['methods'];mapping={x['method']:x for x in methods}
freqonly_metrics={x['method']:x for x in freqonly['metrics']}
rows=[]
for label,correct in [('Full SC-CL-9',r['full_correct']),('均匀随机-9',r['random9_correct']),('Profile-only-9',mapping['Profile-only-9']['correct']),('Violation-only-9',mapping['Violation-only-9']['correct']),('Full配额匹配随机-9',r['frequency_matched_correct'])]:
    rows.append([label,f'{correct:,}',f'{correct/100:.2f}',f"{(r['full_correct']-correct)/100:+.2f}"])
freqonly_correct=round(freqonly_metrics['frequency_only']['direct_em_percent']*freqonly['n']/100)
assert freqonly_correct==7057
rows.append(['独立频次比例随机-9',f'{freqonly_correct:,}',f'{freqonly_correct/100:.2f}',f'{(r["full_correct"]-freqonly_correct)/100:+.2f}'])
emit('fb_controls.tex','FB15K237-2候选来源对照（10,000题，直接解码EM，seed=2026）。各组使用同一Stage~1、训练QA、合法池、$K=9$、损失与优化设置，固定32步并保持120步调度。独立频次组按训练边数成比例无放回抽样；配额匹配组继承Full逐题频率桶配额。$\\Delta$为Full减该行。结果限于单种子和Full选定配置。','tab:fb-controls','lrrr','候选来源 & 正确数 & EM (\\%) & $\\Delta$ (pp)',rows)
comparisons=[('Full $-$ 均匀随机',rand['paired_counts'],rand['net_correct']),('Full $-$ Profile-only',comp['comparisons']['full_minus_profile_only']['paired_counts'],comp['comparisons']['full_minus_profile_only']['net_correct']),('Full $-$ Violation-only',comp['comparisons']['full_minus_violation_only']['paired_counts'],comp['comparisons']['full_minus_violation_only']['net_correct']),('Full $-$ 配额匹配随机',r['full_minus_frequency_matched']['paired_counts'],r['full_minus_frequency_matched']['net_correct']),('配额匹配 $-$ 均匀随机',r['frequency_matched_minus_random9']['paired_counts'],r['frequency_matched_minus_random9']['net_correct'])]
for key,label in [('full_minus_frequency_only','Full $-$ 独立频次'),('frequency_matched_minus_frequency_only','配额匹配 $-$ 独立频次'),('random9_minus_frequency_only','均匀随机 $-$ 独立频次')]:
    item=freqonly['comparisons'][key]['direct']
    comparisons.append((label,item['paired_counts'],item['net_correct']))
emit('fb_paired.tex','FB同10,000题上的配对变化。纠错表示左侧方法正确、右侧错误，退化为相反情况；净变化为纠错减退化。','tab:fb-paired','lrrr','比较方向 & 纠错 & 退化 & 净变化',[[label,c['rescue'],c['regression'],f'{net:+d}'] for label,c,net in comparisons])
slices=read('fb_frequency_slices_seed2026/result.json')
rows=[]
for g,label in [('low','低频（$\\leq215$）'),('middle','中频（216--689）'),('high','高频（$>689$）')]:
    m={x['method']:x for x in slices['metrics'] if x['frequency_group']==g};p=next(x for x in slices['paired_transitions'] if x['frequency_group']==g and x['right']=='frequency_matched')
    rows.append([label,m['full']['n'],f"{m['full']['em_percent']:.2f}",f"{m['frequency_matched']['em_percent']:.2f}",p['rescue'],p['regression'],f"{p['net_correct']:+d}"])
emit('fb_frequency_slices.tex','按目标关系训练边数划分的FB描述性配对分析。边界为训练关系词表频次的三分位次序统计量，同频次不拆分；未按测试表现调边界。EM单位为\\%；配额匹配指Full配额匹配随机。所有分组均报告，净变化合计为7题，不构成跨种子或因果证据。','tab:fb-frequency-slices','lrrrrrr','目标关系频率 & $n$ & Full EM & 配额匹配EM & 纠错 & 退化 & 净变化',rows)
macro_rows=[]
for method,label in [('random9','均匀随机-9'),('profile_only','Profile-only-9'),('violation_only','Violation-only-9'),('frequency_matched','Full配额匹配随机-9'),('full','Full SC-CL-9')]:
    macro_rows.append([label]+[f"{next(x for x in slices['metrics'] if x['frequency_group']==g and x['method']==method)['macro_relation_em_percent']:.2f}" for g in ('low','middle','high','all')])
emit('fb_macro.tex','FB五方法的按关系等权平均EM（\\%）。分别在各组测试中出现的关系内等权；全体一列对全部215个测试关系等权，而不是对三组均值等权。低、中、高频组分别包含67、72和76个测试关系。','tab:fb-macro','lrrrr','方法 & 低频 & 中频 & 高频 & 全体关系',macro_rows)
rerank=read('fb_matched_rerank_seed2026/result.json')
ks=read('fb_k_sensitivity_seed2026/result.json')
assert rerank['status']=='completed' and len(rerank['metrics'])==3
assert [x['k'] for x in ks['metrics']]==[3,9,15]
labels={'full':'Full SC-CL-9','frequency_matched':'Full配额匹配随机-9','random9':'均匀随机-9','frequency_only':'独立频次比例随机-9'}
rerank_rows=[[labels[x['method']],f"{x['h1_percent']:.2f}",f"{x['mrr_percent']:.4f}",f"{x['h3_percent']:.2f}"] for x in rerank['metrics']]
fm=freqonly_metrics['frequency_only']
rerank_rows.append([labels['frequency_only'],f"{fm['h1_percent']:.2f}",f"{fm['mrr_percent']:.4f}",f"{fm['h3_percent']:.2f}"])
emit('fb_rerank.tex','FB同题同十选项的完整候选重排（\\%，seed=2026，$n=10{,}000$）。Full、配额匹配和均匀随机复用既有固定模型；独立频次组采用对应固定配置模型。无新增选模；直接解码结果另列于表~\\ref{tab:fb-controls}。','tab:fb-rerank','lrrr','候选来源 & H@1 & MRR & H@3',rerank_rows)
emit('fb_k_sensitivity.tex','FB训练负例数的排序敏感性（\\%，seed=2026）。同10,000题及原十选项，K9复用；各K保持相同优化步数，非等计算量。','tab:fb-k-sensitivity','rrrr','训练负例$K$ & H@1 & MRR & H@3',[[x['k'],f"{x['h1_percent']:.2f}",f"{x['mrr_percent']:.4f}",f"{x['h3_percent']:.2f}"] for x in ks['metrics']])
cross_sources=['nell23k_source_controls_seed2026/result.json','wn18rr_source_controls_seed2026/result.json']
cross_rows=[]
paired_rows=[]
pair_labels={'full_minus_random9':'Full $-$ 均匀随机','full_minus_frequency_matched':'Full $-$ 配额匹配','frequency_matched_minus_random9':'配额匹配 $-$ 均匀随机'}
for dataset,name,n,options in [('NELL23K',cross_sources[0],4943,10),('WN18RR',cross_sources[1],3134,11)]:
    result=read(name)
    assert result['status']=='completed' and result['seed']==2026
    observed={x['method']:x for x in result['metrics']}
    assert set(observed)=={'full','random9','frequency_matched'}
    for i,method in enumerate(('random9','frequency_matched','full')):
        item=observed[method]
        assert item['n']==n and item['candidate_count']==options
        cross_rows.append([dataset if i==0 else '',labels[method],*[f"{item[key]:.2f}" for key in ('direct_em_percent','h1_percent','mrr_percent','h3_percent')]])
    for i,(key,label) in enumerate(pair_labels.items()):
        item=result['comparisons'][key]
        direct=item['direct_left_minus_right'];ranking=item['ranking']
        assert direct['n']==n and ranking['n']==n
        dc=direct['paired_counts'];rc=ranking['paired_counts']
        assert dc['rescue']-dc['regression']==direct['net_correct']
        assert rc['rescue']-rc['regression']==ranking['net_correct']
        paired_rows.append([dataset if i==0 else '',label,f"{dc['rescue']}/{dc['regression']}",f"{direct['delta_em_pp']:+.2f}",f"{rc['rescue']}/{rc['regression']}",f"{ranking['delta_h1_pp']:+.2f}"])
emit('cross_dataset_sources.tex','NELL23K与WN18RR的三来源完整测试结果（\\%，seed=2026）。NELL共4,943题、原10选项；WN共3,134题、原11选项。各图复用历史获选Full，新增两个固定配置控制，不独立重新调参。WN为历史候选池条件下的探索性比较，实体对核验与B1复现限制见附录~\\ref{sec:wn-exploratory}；这里的MRR为原选项内重排指标。','tab:cross-dataset-sources','llrrrr','数据集 & 候选来源 & 直接EM & H@1 & MRR & H@3',cross_rows)
emit('cross_source_pairs.tex','跨图三来源的逐题配对变化（seed=2026）。纠错/退化均按左侧方法减右侧方法计算；$\\Delta$单位为百分点。直接EM与重排H@1分别计数，不将不同答案选择规则混为一种指标；不提供跨种子不确定性估计。','tab:cross-source-pairs','llrrrr','数据集 & 比较方向 & 直接纠错/退化 & $\\Delta$ EM & 排序纠错/退化 & $\\Delta$ H@1',paired_rows)
manifest={str(PAPER/'artifacts'/name):hashlib.sha256((PAPER/'artifacts'/name).read_bytes()).hexdigest() for name in source_names+['fb_frequency_matched_control_seed2026/result.json','fb_component_ablation_seed2026/result.json','fb_random9_control_seed2026/result.json','fb_frequency_only_control_seed2026/result.json','fb_frequency_only_control_seed2026/metrics.csv','fb_frequency_only_control_seed2026/paired_all_methods.csv','fb_frequency_only_control_seed2026/selected_candidates.csv','fb_frequency_only_control_seed2026/preparation_audit.json','fb_frequency_only_control_seed2026/evidence_index.json','fb_frequency_slices_seed2026/result.json','fb_matched_rerank_seed2026/result.json','fb_k_sensitivity_seed2026/result.json']+cross_sources}
(PAPER/'artifacts/table_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('Built nine evidence-backed manuscript tables.')

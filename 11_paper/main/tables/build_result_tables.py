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
r=read('fb_frequency_matched_control_seed2026/result.json');comp=read('fb_component_ablation_seed2026/result.json');rand=read('fb_random9_control_seed2026/result.json')
methods=comp['methods'];mapping={x['method']:x for x in methods}
rows=[]
for label,correct in [('Full SC-CL-9',r['full_correct']),('均匀随机-9',r['random9_correct']),('Profile-only-9',mapping['Profile-only-9']['correct']),('Violation-only-9',mapping['Violation-only-9']['correct']),('Full配额匹配随机-9',r['frequency_matched_correct'])]:
    rows.append([label,f'{correct:,}',f'{correct/100:.2f}',f"{(r['full_correct']-correct)/100:+.2f}"])
emit('fb_controls.tex','FB15K237-2候选来源对照（10,000题，直接解码EM，seed=2026）。各组使用同一Stage~1、训练QA、合法池、$K=9$、损失与优化设置，固定32步并保持120步调度。$\\Delta$为Full减该行。配额匹配组继承Full逐题频率桶配额，并非独立通用频率采样器；本表不提供跨种子不确定性或统计等效结论。','tab:fb-controls','lrrr','候选来源 & 正确数 & EM (\\%) & $\\Delta$ (pp)',rows)
comparisons=[('Full $-$ 均匀随机',rand['paired_counts'],rand['net_correct']),('Full $-$ Profile-only',comp['comparisons']['full_minus_profile_only']['paired_counts'],comp['comparisons']['full_minus_profile_only']['net_correct']),('Full $-$ Violation-only',comp['comparisons']['full_minus_violation_only']['paired_counts'],comp['comparisons']['full_minus_violation_only']['net_correct']),('Full $-$ 配额匹配随机',r['full_minus_frequency_matched']['paired_counts'],r['full_minus_frequency_matched']['net_correct']),('配额匹配 $-$ 均匀随机',r['frequency_matched_minus_random9']['paired_counts'],r['frequency_matched_minus_random9']['net_correct'])]
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
manifest={str(PAPER/'artifacts'/name):hashlib.sha256((PAPER/'artifacts'/name).read_bytes()).hexdigest() for name in source_names+['fb_frequency_matched_control_seed2026/result.json','fb_component_ablation_seed2026/result.json','fb_random9_control_seed2026/result.json','fb_frequency_slices_seed2026/result.json']}
(PAPER/'artifacts/writing_revision_20261106/table_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('Built five evidence-backed manuscript tables.')

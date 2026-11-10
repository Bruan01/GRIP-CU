"""Portable literature-guided plots from audited CSV exports (CPU only)."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image, ImageOps
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / 'artifacts/sampling_diagnostics_seed2026'
DS = ['nell23k','codexm','fb15k237_2','wn18rr']
NAMES = dict(zip(DS,['NELL23K','CoDEx-M','FB15K237-2','WN18RR']))
LABELS = {'full':'Full SC-CL','legal_uniform_expected':'Uniform (expected)',
          'uniform_observed':'Uniform (observed)','quota_observed':'Quota-matched',
          'profile_only_observed':'Profile only','violation_only_observed':'Violation only'}
COLORS = {'full':'#427C96','legal_uniform_expected':'#505A5E',
          'uniform_observed':'#899795','quota_observed':'#9B859E',
          'profile_only_observed':'#AD9A7D','violation_only_observed':'#8497B2'}
STYLES = {'full':'-','legal_uniform_expected':(0,(1,2)),
          'uniform_observed':(0,(5,2)),'quota_observed':(0,(3,2,1,2))}
ORDER = ['legal_uniform_expected','uniform_observed','quota_observed','full']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,
    'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'legend.fontsize':6.5,'axes.spines.right':False,'axes.spines.top':False,
    'axes.linewidth':.65,'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','svg.hashsalt':'sccl_diagnostics_seed2026',
    'figure.facecolor':'white','savefig.facecolor':'white','legend.frameon':False})
EDGES = np.linspace(0,1,11)
CMAP = LinearSegmentedColormap.from_list('muted_lift',['#557E95','#F4F3EF','#AE776F'])
CMAP.set_bad('#D9DCDC')
QA = {}

def read(name):
    with (DATA / 'source_data' / name).open(newline='') as f:
        return list(csv.DictReader(f))

HIST = defaultdict(lambda:defaultdict(dict))
for r in read('histograms.csv'):
    HIST[r['dataset']][r['feature']].setdefault(r['method'],[]).append(float(r['probability']))
GRID = defaultdict(dict)
for r in read('joint_grid.csv'):
    GRID[r['dataset']].setdefault(r['method'],np.zeros((10,10)))[int(r['structure_bin']),int(r['hardness_bin'])] = float(r['probability'])
REL = defaultdict(lambda:defaultdict(list))
for r in read('relation_distribution.csv'):
    REL[r['dataset']][r['method']].append(float(r['probability']))
QUESTIONS = read('questions.csv')
SUMMARY = {(r['dataset'],r['method']):r for r in read('method_summary.csv')}

def title(ax,text,letter):
    ax.set_title(text,loc='left',pad=8)
    ax.text(-.12,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=9)
    ax.tick_params(length=3,width=.65)

def histogram(ax,d,feature,ylabel=True):
    for m in ORDER:
        if m not in HIST[d][feature]: continue
        ax.stairs(np.array(HIST[d][feature][m])*100,EDGES,color=COLORS[m],
                  linestyle=STYLES[m],linewidth=1.9 if m=='full' else 1.1,zorder=4 if m=='full' else 2)
    ax.set(xlim=(0,1),ylabel='Negative slots (%)' if ylabel else '')
    ax.set_xticks([0,.5,1]); ax.set_ylim(bottom=0)
    ax.grid(axis='y',color='#E9ECEC',lw=.5); ax.set_axisbelow(True)

def legend(fig,y,methods=None):
    methods=ORDER if methods is None else methods
    fig.legend(handles=[Line2D([0],[0],label=LABELS[m],color=COLORS[m],linestyle=STYLES[m],lw=1.7) for m in methods],
               loc='upper center',bbox_to_anchor=(.5,y),ncol=4,columnspacing=1.2,handlelength=2.6)

def heat(ax,d):
    f,u = GRID[d]['full'], GRID[d]['legal_uniform_expected']
    with np.errstate(divide='ignore',invalid='ignore'):
        lift = np.log2(f/u)
    # Mask only impossible cells; zero-observation cells remain lower-extension color.
    z = np.ma.array(np.clip(lift,-2.01,2.01),mask=u==0)
    im = ax.pcolormesh(EDGES,EDGES,z.T,cmap=CMAP,norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2),rasterized=False)
    for x,y in np.ndindex(10,10):
        if u[x,y]==0:
            ax.text((x+.5)/10,(y+.5)/10,'×',ha='center',va='center',fontsize=5,color='#626B6D')
        elif abs(lift[x,y])>=.5:
            ax.text((x+.5)/10,(y+.5)/10,'+' if lift[x,y]>0 else '−',ha='center',va='center',fontsize=5,color='#35484E')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Structure percentile',ylabel='B1 hardness percentile')
    ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1]); ax.set_aspect('equal')
    return im

def export(fig,name,book):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    outside=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text(): continue
        bb=t.get_window_extent(renderer)
        if bb.width and bb.height and (bb.x0 < -2 or bb.y0 < -2 or bb.x1 > fig.bbox.width+2 or bb.y1 > fig.bbox.height+2):
            outside.append(t.get_text())
    assert not outside,(name,outside)
    for ext in ['pdf','svg','png']:
        fig.savefig(HERE / f'{name}.{ext}',dpi=230)
    book.savefig(fig)
    svg=ET.parse(HERE / f'{name}.svg').getroot()
    text_count=len(svg.findall('.//{http://www.w3.org/2000/svg}text'))
    assert text_count>5 and len(PdfReader(HERE / f'{name}.pdf').pages)==1
    QA[name]={'text_outside_canvas':outside,'editable_svg_text_count':text_count,
              'pdf_pages':1,'size_inches':list(fig.get_size_inches())}
    plt.close(fig)

def main():
    with PdfPages(HERE / 'sampling_diagnostics_all.pdf') as book:
        fig,axes=plt.subplots(2,4,figsize=(7.2,4.2))
        fig.subplots_adjust(left=.075,right=.985,bottom=.17,top=.83,wspace=.3,hspace=.65)
        for j,d in enumerate(DS):
            histogram(axes[0,j],d,'hardness_percentile',j==0)
            histogram(axes[1,j],d,'structure_percentile',j==0)
            title(axes[0,j],NAMES[d],chr(97+j)); title(axes[1,j],NAMES[d],chr(101+j))
            axes[0,j].set_xlabel('B1 hardness percentile')
            axes[1,j].set_xlabel('Structure percentile')
        # Match limits within each row so effect sizes remain comparable.
        for row in axes:
            ymax=max(a.get_ylim()[1] for a in row)
            for a in row: a.set_ylim(0,ymax)
        legend(fig,.985)
        fig.text(.5,.045,'K = 9; seed = 2026. Within-query midranks; higher = harder / more similar.\nCoDEx-M: expected Uniform only. WN18RR: historical pool, including 55 unverified pairs.',ha='center',fontsize=6.3)
        export(fig,'fig01_selection_distributions',book)

        fig,axes=plt.subplots(1,4,figsize=(7.2,2.9))
        fig.subplots_adjust(left=.075,right=.985,bottom=.42,top=.84,wspace=.36)
        for j,d in enumerate(DS):
            im=heat(axes[j],d); title(axes[j],NAMES[d],chr(97+j))
            if j: axes[j].set_ylabel('')
        cbax=fig.add_axes([.32,.20,.4,.035])
        fig.colorbar(im,cax=cbax,orientation='horizontal',extend='both',ticks=[-2,-1,0,1,2],label='log2 selection lift: Full / expected Uniform')
        fig.text(.5,.018,'Gray ×: no eligible mass. Signs: |log2 lift| ≥ 0.5. Zero Full mass uses the lower extension.\nWN18RR retains historical pools; selection enrichment is descriptive.',ha='center',fontsize=6.1)
        export(fig,'fig02_joint_selection_lift',book)

        fig,axes=plt.subplots(2,2,figsize=(7.2,5.8))
        fig.subplots_adjust(left=.1,right=.98,bottom=.13,top=.9,wspace=.3,hspace=.55)
        for j,(d,ax) in enumerate(zip(DS,axes.flat)):
            x=np.array(REL[d]['train_edges']); y=np.array(REL[d]['full'])
            assert np.all(x>0)
            pos=y>0; low=min(x.min(),y[pos].min())*.7; high=max(x.max(),y.max())*1.3
            ax.plot([low,high],[low,high],color='#6B7173',linestyle='--',lw=1,label='Equal marginal probability')
            ax.scatter(x[pos],y[pos],s=12,color=COLORS['full'],alpha=.65,edgecolors='none',label='Full SC-CL')
            if (~pos).any():
                ax.scatter(x[~pos],np.repeat(low,(~pos).sum()),marker='v',s=18,facecolors='none',edgecolors=COLORS['full'],label='Zero sampled mass (at floor)')
            ax.set(xscale='log',yscale='log',xlim=(low,high),ylim=(low,high),xlabel='Training-edge relation probability',ylabel='Sampled-negative relation probability')
            ticks=10.**np.arange(np.ceil(np.log10(low)),np.floor(np.log10(high))+1)
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
            title(ax,NAMES[d],chr(97+j)); ax.grid(color='#ECEEEE',lw=.5); ax.set_axisbelow(True)
            ax.legend(loc='upper left',fontsize=6)
        fig.text(.5,.035,'Each dot is one mapped relation label. Both axes use log scales.\nThe equality line is descriptive; no power-law fit or original-distribution matching objective is assumed.',ha='center',fontsize=6.3)
        export(fig,'fig03_relation_marginals',book)

        fig,axes=plt.subplots(1,2,figsize=(7.2,2.9))
        fig.subplots_adjust(left=.18,right=.97,bottom=.25,top=.81,wspace=.4)
        methods=['uniform_observed','quota_observed','profile_only_observed','violation_only_observed','full']
        for j,(ax,feature) in enumerate(zip(axes,['structure_percentile','hardness_percentile'])):
            ax.axvline(.5,color=COLORS['legal_uniform_expected'],linestyle=':',lw=1.1)
            for i,m in enumerate(methods):
                val=float(SUMMARY['fb15k237_2',m][feature])
                ax.scatter(val,i,s=38 if m=='full' else 27,color=COLORS[m],marker='D' if m=='full' else 'o',zorder=4)
                ax.text(val+.002,i,f'{val:.3f}',va='center',fontsize=6.5)
            ax.set_yticks(range(5),[LABELS[m] for m in methods] if j==0 else [])
            ax.invert_yaxis(); ax.set(xlim=(.49,.599),xlabel='Mean within-query percentile')
            ax.set_xticks([.50,.525,.55,.575]); ax.grid(axis='x',color='#ECEEEE',lw=.5)
            title(ax,'Structure similarity' if j==0 else 'Frozen B1 hardness',chr(97+j))
        fig.suptitle('FB15K237-2 | completed candidate-source controls',fontsize=10,y=.98)
        fig.text(.5,.055,'Dotted line: exact expected Uniform mean (0.5). Dots: actual frozen banks; K = 9, seed = 2026.\nThese are selection diagnostics, without seed uncertainty or causal performance attribution.',ha='center',fontsize=6.3)
        export(fig,'fig04_fb_component_fingerprints',book)

        for d in DS:
            fig,axes=plt.subplots(2,2,figsize=(7.2,5.8))
            fig.subplots_adjust(left=.095,right=.94,bottom=.14,top=.82,wspace=.35,hspace=.65)
            histogram(axes[0,0],d,'structure_raw'); title(axes[0,0],'Relation-profile similarity','a')
            axes[0,0].set_xlabel('Raw structure similarity')
            histogram(axes[0,1],d,'hardness_raw'); title(axes[0,1],'Frozen B1 difficulty','b')
            axes[0,1].set_xlabel('Cached pairwise confusion')
            im=heat(axes[1,0],d); title(axes[1,0],'Joint selection lift','c')
            cbax=fig.add_axes([.445,.20,.012,.19])
            cb=fig.colorbar(im,cax=cbax,extend='both',ticks=[-2,0,2]); cb.set_label('log2 lift',labelpad=2)
            ax=axes[1,1]
            qs=[r for r in QUESTIONS if r['dataset']==d]
            controls=['uniform_observed','quota_observed'] if d!='codexm' else ['legal_uniform_expected']
            for m in controls:
                delta=np.sort([float(r['full_hardness_percentile'])-float(r[m+'_hardness_percentile']) for r in qs])
                assert delta.min()>=-.6 and delta.max()<=.6, 'ECDF range must retain every question'
                ax.step(np.r_[-.6,delta,.6],np.r_[0,np.arange(1,len(delta)+1)/len(delta),1],where='post',color=COLORS[m],linestyle=STYLES[m],label='Full minus '+LABELS[m])
            ax.axvline(0,color='#777C7E',lw=.7)
            ax.set(xlim=(-.6,.6),ylim=(0,1.02),xlabel='Paired mean hardness percentile difference',ylabel='Fraction of questions')
            ax.set_xticks([-.6,0,.6]); ax.grid(axis='y',color='#ECEEEE',lw=.5)
            ax.legend(loc='lower right',fontsize=6); title(ax,'Question-level paired differences','d')
            legend(fig,.99,[m for m in ORDER if m in HIST[d]['structure_raw']]); fig.suptitle(NAMES[d]+' | actual training-negative diagnostics',y=.90,fontsize=10)
            scope='WN18RR retains 55 unverified historical entity pairs.' if d=='wn18rr' else ('Uniform is analytic only; no trained CoDEx-M control is implied.' if d=='codexm' else 'Uniform and quota-matched curves use the completed, aligned control banks.')
            fig.text(.5,.035,'K = 9; seed = 2026. Each question contributes equal mass.\n'+scope+'\nFrozen B1 scores describe candidate generation; they are not final SC-CL predictions.',ha='center',fontsize=6.3)
            export(fig,'diagnostics_'+d,book)
    assert len(PdfReader(HERE/'sampling_diagnostics_all.pdf').pages)==8
    thumbs=[]
    for name in QA:
        with Image.open(HERE/f'{name}.png') as im:
            im=im.convert('RGB'); im.thumbnail((900,730)); thumbs.append(im.copy())
    row_heights=[max(im.height for im in thumbs[i:i+2])+20 for i in range(0,len(thumbs),2)]
    contact=Image.new('RGB',(1820,sum(row_heights)),'white')
    for i,im in enumerate(thumbs):
        contact.paste(im,((i%2)*920,sum(row_heights[:i//2])))
    contact.save(HERE/'overview.png'); ImageOps.grayscale(contact).save(HERE/'overview_grayscale.png')
    QA['_package']={'combined_pdf_pages':8,'source_data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((DATA/'source_data').glob('*.csv'))}}
    (HERE/'qa_report.json').write_text(json.dumps(QA,indent=2)+'\n')
    print('Exported eight figure sets and combined PDF',flush=True)

if __name__=='__main__': main()

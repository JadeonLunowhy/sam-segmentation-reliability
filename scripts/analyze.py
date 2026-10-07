"""Rebuild every statistic and plot from real saved observations."""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src.data import ROOT,Case,load_case,sha256_file
from src.metrics import summarize,selective_curve,aurc
from src.statistics import bootstrap_difference

METHODS=['confidence','stability','consistency','minimum','fusion','flip']
LABELS={'confidence':'SAM confidence','stability':'Threshold stability','consistency':'Prompt consistency',
        'minimum':'Minimum consistency','fusion':'Confidence + consistency','flip':'Flip consistency'}

def flatten(rows,cell):
    return [dict(r,**{k:v for k,v in r['sweep'][cell].items() if k not in ['confidence']}) for r in rows]

def evaluate(rows):
    q=np.array([r['true_iou'] for r in rows])
    return {m:dict(summarize(q,np.array([r[m] for r in rows])),
                   spearman=float(spearmanr(q,[r[m] for r in rows]).statistic) if len(set(r[m] for r in rows))>1 and len(set(q))>1 else None) for m in METHODS}

def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def main():
    config=json.loads((ROOT/'configs/study.json').read_text());frozen=json.loads((ROOT/'configs/frozen.json').read_text());cell=frozen['selected']
    results={};boot={};datasets={}
    for split in ['test','extra']:
        path=ROOT/'results'/(split+'.jsonl'); raw=[json.loads(l) for l in path.read_text().splitlines()];rows=flatten(raw,cell);datasets[split]=rows
        q=np.array([r['true_iou'] for r in rows]); stats=evaluate(rows)
        groups={regime:evaluate([r for r in rows if r['regime']==regime]) for regime in config['regimes']}
        valid_rows=[r for r in rows if r['outside'][cell]==0]
        size_groups={name:evaluate(sub) for name,sub in [('small',[r for r in rows if r['object_fraction']<0.2]),('large',[r for r in rows if r['object_fraction']>=0.2])] if sub}
        conf_groups={name:evaluate(sub) for name,sub in [('below_0.9',[r for r in rows if r['confidence']<0.9]),('at_least_0.9',[r for r in rows if r['confidence']>=0.9])] if sub}
        results[split]={'images':len({r['image_id'] for r in rows}),'cases':len(rows),'mean_iou':float(q.mean()),'methods':stats,'by_regime':groups,
                        'all_perturbations_inside':{'cases':len(valid_rows),'methods':evaluate(valid_rows) if valid_rows else None},
                        'by_size':size_groups,'by_confidence':conf_groups,'source_sha256':sha256_file(path),
                        'stable_errors':sum(r['consistency']>=0.9 and r['true_iou']<0.5 for r in rows),
                        'mean_outside_fraction':float(np.mean([r['outside'][cell] for r in rows]))}
        boot[split]={}
        for method in ['consistency','fusion','flip']:
            for metric in ['aurc','auroc']:
                boot[split][method+'_'+metric]=bootstrap_difference(rows,method,'confidence',metric,config['bootstrap_draws'],config['seed'])
        fig,ax=plt.subplots(figsize=(7,4.2))
        records=[]
        for method in METHODS:
            coverage,risk=selective_curve(q,np.array([r[method] for r in rows]));ax.plot(coverage,risk,label=LABELS[method],lw=1.8)
            records.extend({'method':method,'coverage':float(c),'risk':float(r)} for c,r in zip(coverage,risk))
        ax.axhline(1-q.mean(),ls='--',color='gray',label='Random expectation')
        coverage,risk=selective_curve(q,q);ax.plot(coverage,risk,':',color='black',label='Oracle (uses labels)')
        ax.set(xlabel='Retained fraction (coverage)',ylabel='Mean retained error (1 - IoU)',title=('Main test' if split=='test' else 'Additional Pet corpus')+' | '+cell)
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'figures'/(split+'_risk.png'),dpi=180);plt.close(fig)
        with open(ROOT/'figures'/(split+'_risk.csv'),'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['method','coverage','risk']);w.writeheader();w.writerows(records)
    write_json(ROOT/'results/summary.json',{'selected':cell,'datasets':results,'failure_thresholds':[0.5,0.75]})
    write_json(ROOT/'results/bootstrap.json',boot)
    # Registered test sweep is sensitivity analysis, never a test-driven selection.
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));sensitivity={}
    for ax,split in zip(axs,['test','extra']):
        rows=datasets[split];q=np.array([r['true_iou'] for r in rows]);matrix=[]
        for radius in config['radii']:
            matrix.append([aurc(q,np.array([r['sweep'][f'r{radius:g}_k{k}']['fusion'] for r in rows])) for k in config['budgets']])
        sensitivity[split]=matrix;im=ax.imshow(matrix,cmap='viridis_r');ax.set_xticks(range(3),config['budgets']);ax.set_yticks(range(3),['1%','3%','6%']);ax.set(xlabel='Additional prompt decodes K',ylabel='Jitter radius / image diagonal',title=split+' fusion AURC')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i][j]:.3f}',ha='center',va='center',color='white',fontsize=12)
        fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(ROOT/'figures/sensitivity.png',dpi=180);plt.close(fig);write_json(ROOT/'figures/sensitivity.json',sensitivity)
    timing=json.loads((ROOT/'results/budget_timing.json').read_text())
    cost_rows=[];fig,ax=plt.subplots(figsize=(7,4))
    radius=float(cell[1:].split('_k')[0]);main_rows=datasets['test'];q=np.array([r['true_iou'] for r in main_rows])
    entries=[('Confidence',0,results['test']['methods']['confidence']['aurc'])]
    for k in config['budgets']:
        seconds=float(np.mean([r['additional_prompt_seconds'][str(k)] for r in timing['samples']]))
        risk=aurc(q,np.array([r['sweep'][f'r{radius:g}_k{k}']['fusion'] for r in main_rows]))
        entries.append((f'Fusion K={k}',seconds,risk))
    entries.append(('Flip',float(np.mean([r['additional_flip_seconds'] for r in timing['samples']])),results['test']['methods']['flip']['aurc']))
    for name,seconds,risk in entries:
        ax.scatter(seconds,risk,s=70);ax.annotate(name,(seconds,risk),xytext=(5,6),textcoords='offset points')
        cost_rows.append({'method':name,'additional_seconds':seconds,'test_aurc':risk})
    ax.set(xlabel='Measured additional inference seconds / case',ylabel='Main-test AURC (lower is better)',title=f'Cost and ranking quality | radius {radius:g}')
    ax.margins(.15);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'figures/compute_quality.png',dpi=180);plt.close(fig)
    write_json(ROOT/'figures/compute_quality.json',{'timing_source':'results/budget_timing.json','evaluation_source':'results/test.jsonl','points':cost_rows})
    cases={c['image_id']:Case(**c) for c in json.loads((ROOT/'data/splits.json').read_text())['cases']}
    rows=datasets['test'];stable=[r for r in rows if r['consistency']>=.9 and r['true_iou']<.5]
    candidates=sorted(stable,key=lambda r:r['true_iou'])[:2]
    remaining=sorted([r for r in rows if r not in candidates],key=lambda r:r['true_iou'])
    candidates+=remaining[:max(0,3-len(candidates))]
    candidates+=sorted(rows,key=lambda r:-r['true_iou'])[:1]
    fig,axs=plt.subplots(len(candidates),3,figsize=(14,len(candidates)*2.4),squeeze=False)
    for i,row in enumerate(candidates):
        image,target,valid=load_case(cases[row['image_id']]);pred=np.load(ROOT/row['mask_path'])['original']
        for j,(mask,title) in enumerate([(None,'Input + simulated click'),(target,'Reference foreground'),(pred,'SAM prediction')]):
            ax=axs[i,j];ax.imshow(image)
            if mask is not None:ax.imshow(np.ma.masked_where(~mask,mask),cmap='spring',alpha=.5,vmin=0,vmax=1)
            ax.plot(*row['point'],'+',color='cyan',ms=12,mew=2);ax.axis('off')
            ax.set_title(title if i==0 else '',fontsize=10)
        axs[i,0].set_ylabel(row['image_id'])
        axs[i,2].set_title(f"IoU {row['true_iou']:.2f} | confidence {row['confidence']:.2f} | consistency {row['consistency']:.2f}",fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'figures/examples.png',dpi=170);plt.close(fig)
    write_json(ROOT/'figures/examples.json',[{k:r[k] for k in ['image_id','regime','true_iou','confidence','consistency']} for r in candidates])
    # Presentation-specific crop: two failures with large readable labels.
    show=candidates[:2];fig,axs=plt.subplots(len(show),3,figsize=(14,5.5),squeeze=False)
    for i,row in enumerate(show):
        image,target,valid=load_case(cases[row['image_id']]);pred=np.load(ROOT/row['mask_path'])['original']
        for j,(mask,title) in enumerate([(None,'Image + click'),(target,'Reference target'),(pred,'SAM prediction')]):
            ax=axs[i,j];ax.imshow(image)
            if mask is not None:ax.imshow(np.ma.masked_where(~mask,mask),cmap='spring',alpha=.5,vmin=0,vmax=1)
            ax.plot(*row['point'],'+',color='cyan',ms=12,mew=2);ax.axis('off')
            ax.set_title(title if i==0 else '',fontsize=16)
        axs[i,2].set_title(f"IoU {row['true_iou']:.2f} | consistency {row['consistency']:.2f}",fontsize=16)
    fig.tight_layout();fig.savefig(ROOT/'figures/examples_slides.png',dpi=170);plt.close(fig)
    print(json.dumps({'selected':cell,'test':results['test']['methods']['fusion'],'extra':results['extra']['methods']['fusion']},indent=2))

if __name__=='__main__':main()

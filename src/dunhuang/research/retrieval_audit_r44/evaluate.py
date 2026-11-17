"""Train-only scale, frozen weights, same-population negative controls and group summaries."""
import argparse,collections
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.sdmr_event_rank.scorer import scores,retrieval_metrics
from dunhuang.research.sdmr_event_rank.evaluate import group_ci
from .path import partial

METHODS=('Energy','Global','Diagonal','Offset','Partial','CalibratedPartial','TrainBankDebias')

def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh evaluation')
    out.mkdir(parents=True);meta=read(root/'benchmark/manifest.json');fm=read(root/'features/result.json');assert fm['benchmark_sha256']==sha(root/'benchmark/manifest.json');z=Arrays(root/'features/arrays.npz');cfg=meta['config'];rows=meta['rows'];tasks=meta['tasks'];donors=fm['donors'];cache={}
    def token(i,zero=False):
        key=('z' if zero else 'a')+str(i)
        if key not in cache:cache[key]={n:z[key+'_'+n] for n in ('mg','dg','mt','dt','mv','dv','positions')}
        return cache[key]
    # Each real candidate pair counted once per group, not once for each duplicate task.
    evidence=collections.defaultdict(lambda:dict(positive=[],negative=[]));seen=set()
    for t in tasks:
        if t['role']!='train':continue
        q=token(t['query'])
        for i,p in zip(t['candidates'],t['positive']):
            identity=(t['query'],i)
            if identity in seen:continue
            seen.add(identity);c=token(i);mask=q['mv']&c['dv'];v=np.sum(q['mt']*c['dt'],axis=-1)[mask][::cfg['token_stride']];evidence[t['source_group']]['positive' if p else 'negative'].extend(v.tolist())
    quantiles={g:float(np.quantile(v['negative'],.95)) for g,v in evidence.items() if g in meta['calibration_groups'] and v['negative']};threshold=float(np.median(list(quantiles.values())));drop=float(np.clip((1-threshold)/2,.05,.75));probe={}
    for g,v in evidence.items():
        if g not in meta['probe_groups']:continue
        a=np.asarray(v['positive']);b=np.asarray(v['negative']);probe[g]=dict(positive_tokens=len(a),negative_tokens=len(b),tpr=float(np.mean(a>threshold)) if len(a) else None,fpr=float(np.mean(b>threshold)) if len(b) else None)
    supported=[v for v in probe.values() if v['tpr'] is not None and v['fpr'] is not None];fpr=float(np.mean([v['fpr'] for v in supported]));tpr=float(np.mean([v['tpr'] for v in supported]));calibration=dict(group_negative_q95=quantiles,threshold=threshold,drop_cost=drop,probe=probe,probe_macro_fpr=fpr,probe_macro_tpr=tpr,informative=bool(fpr<=.1 and tpr>2*fpr+.05),scope='TRAIN_ONLY_NEGATIVE_TOKEN_SCALE_NOT_SEMANTIC_NULL_PROBABILITY')
    write(out/'calibration.json',calibration);banks={}
    for length in cfg['lengths']:
        ids=[]
        for group in meta['calibration_groups']:
            opts=sorted({t['query'] for t in tasks if t['source_group']==group and t['frames']==length})
            if opts:ids.append(opts[0])
        banks[length]=ids
    mu=np.array(meta['energy_calibration']['mean']);sd=np.array(meta['energy_calibration']['std']);arrays={};records=[];affinity={}
    def bank(i,zero):
        key=(i,zero)
        if key not in affinity:
            ids=[j for j in banks[rows[i]['frames']] if rows[j]['audio_sha256']!=rows[i]['audio_sha256']];v=np.array([token(j)['mg']@token(i,zero)['dg'] for j in ids]);affinity[key]=float(np.sort(v)[-min(10,len(v)):].mean())
        return affinity[key]
    for control in ('actual','both_zero','music_swap','motion_swap'):
        for k,t in enumerate(tasks):
            qi=donors[str(t['query'])] if control=='music_swap' else t['query'];q=token(qi,control=='both_zero');values=[];coverage=[]
            for i in t['candidates']:
                ci=donors[str(i)] if control=='motion_swap' else i;c=token(ci,control=='both_zero');s,_=scores(q,c,cfg,False);p,pc,_=partial(q,c,cfg);cp,cc,_=partial(q,c,cfg,drop);qm=0 if control=='both_zero' else rows[qi]['music_scalar'];dm=0 if control=='both_zero' else rows[ci]['motion_scalar'];energy=-abs((qm-mu[0])/sd[0]-(dm-mu[1])/sd[1]);values.append([energy,s['Global'],s['Diagonal'],s['Offset'],p,cp,2*s['Global']-bank(ci,control=='both_zero')]);coverage.append([pc,cc])
            values=np.asarray(values);arrays[f'{control}_{k}_scores']=values;arrays[f'{control}_{k}_coverage']=np.asarray(coverage);records.append(dict(task=k,role=t['role'],source_group=t['source_group'],pool=t['pool'],frames=t['frames'],control=control,metrics={m:retrieval_metrics(values[:,j],t['positive']) for j,m in enumerate(METHODS)}))
        print('R44_SCORED',control,len(tasks),flush=True)
    summary={}
    for role in ('train','val'):
        summary[role]={}
        for pool in ('same_music','energy_matched','cross_music'):
            summary[role][pool]={}
            for control in ('actual','both_zero','music_swap','motion_swap'):
                part=[r for r in records if r['role']==role and r['pool']==pool and r['control']==control];methods={}
                for m in METHODS:
                    gs=collections.defaultdict(list)
                    for r in part:gs[r['source_group']].append(r['metrics'][m]['recall_at1'])
                    means={g:float(np.mean(v)) for g,v in gs.items()};methods[m]=dict(group_means=means,group_macro_recall_at1=float(np.mean(list(means.values()))) if means else None,cluster_ci95=group_ci(list(means.values()),44041,5000),tasks=len(part))
                summary[role][pool][control]=methods
    np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(summary=summary,records=records,calibration=calibration,train_bank_indices=banks,methods=METHODS,benchmark_sha256=sha(root/'benchmark/manifest.json'),feature_sha256=sha(root/'features/arrays.npz'),checkpoint_sha256=fm['checkpoint_sha256'],same_population_controls=True,bank_scope='FIXED_TRAIN_CALIBRATION_QUERY_BANK_K10; NOT_FULL_BEATDANCE_REPRODUCTION; NO_NATIVE_DEPLOYMENT',optimizer_steps=0,can_route=False,human_reviewed=False,operational_qualified=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)

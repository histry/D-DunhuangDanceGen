"""Frozen equal-population ranking; the training gate never consumes Val metrics."""
import argparse,collections,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.music_reperformance_r43.replay import exact_arrays
from .scorer import scores,batch_partial,retrieval_metrics

METHODS=('Duration','Energy','Global','Diagonal','Offset','Partial','Hybrid')


def group_ci(values,seed,draws):
    x=np.asarray(values,float)
    if len(x)<2:return [None,None]
    rng=np.random.default_rng(seed);v=x[rng.integers(len(x),size=(draws,len(x)))].mean(1);return np.quantile(v,[.025,.975]).tolist()


def run(benchmark,features,out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh frozen ranking evaluation')
    out.mkdir(parents=True);meta=read(Path(benchmark)/'manifest.json');cfg=meta['config'];fm=read(Path(features)/'result.json');assert fm['benchmark_sha256']==sha(Path(benchmark)/'manifest.json');z=np.load(Path(features)/'arrays.npz');rows=meta['rows'];tasks=meta['tasks'];arrays={};result=[];donors=fm['donors'];mu=np.array(meta['energy_calibration']['mean']);sd=np.array(meta['energy_calibration']['std']);cache={}
    def token(i,zero=False):
        key=('z' if zero else 'a')+str(i)
        if key not in cache:cache[key]={n:z[key+'_'+n] for n in ('mg','dg','mt','dt','mv','dv','positions')}
        return cache[key]
    calibration=[]
    # Train-only scale calibration, not fitting coefficients or using target labels.
    for task in tasks:
        if task['role']!='train':continue
        q=token(task['query'])
        for i in task['candidates']:
            s,_=scores(q,token(i),cfg,False);calibration.append([s['Global'],s['Diagonal']])
    calibration=np.array(calibration);means=calibration.mean(0);scales=calibration.std(0).clip(1e-4)
    for control in cfg['controls']:
        for k,task in enumerate(tasks):
            qi=task['query'];ids=task['candidates'];q=token(donors[str(qi)] if control=='music_swap' else qi,control in ('music_zero','both_zero'))
            cs=[token(donors[str(i)] if control=='motion_swap' else i,control=='both_zero') for i in ids];ps,pc,cost=batch_partial(q,cs,cfg);values=[];coverage=[]
            for j,(i,c) in enumerate(zip(ids,cs)):
                s,_=scores(q,c,cfg,False);row=rows[donors[str(i)]] if control=='motion_swap' else rows[i];qm=0. if control in ('music_zero','both_zero') else rows[donors[str(qi)] if control=='music_swap' else qi]['music_scalar'];dm=0. if control=='both_zero' else row['motion_scalar'];energy=-abs((qm-mu[0])/sd[0]-(dm-mu[1])/sd[1]);hybrid=.5*((np.array([s['Global'],s['Diagonal']])-means)/scales).sum();values.append([0.,energy,s['Global'],s['Diagonal'],s['Offset'],float(ps[j]),float(hybrid)]);coverage.append([s['diagonal_coverage'],s['offset_coverage'],float(pc[j]),float(s['selected_offset'])])
            values=np.array(values);arrays[f'{control}_{k}_scores']=values;arrays[f'{control}_{k}_coverage']=np.array(coverage);arrays[f'{control}_{k}_path_cost']=cost;metrics={name:retrieval_metrics(values[:,j],task['positive']) for j,name in enumerate(METHODS)}
            result.append(dict(task=k,uid=task['uid'],role=task['role'],pool=task['pool'],frames=task['frames'],source_group=task['source_group'],control=control,metrics=metrics))
        print('E2_EVALUATED',control,len(tasks),flush=True)
    summary={}
    for role in ('train','val'):
        summary[role]={}
        for pool in ('same_music','energy_matched','cross_music'):
            summary[role][pool]={}
            for control in cfg['controls']:
                part=[r for r in result if r['role']==role and r['pool']==pool and r['control']==control];methods={}
                for name in METHODS:
                    groups=collections.defaultdict(list)
                    for r in part:groups[r['source_group']].append(r['metrics'][name]['recall_at1'])
                    methods[name]=dict(queries=len(part),music_groups=len(groups),group_macro_recall_at1=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None,recall_at1=float(np.mean([r['metrics'][name]['recall_at1'] for r in part])) if part else None,recall_at5=float(np.mean([r['metrics'][name]['recall_at5'] for r in part])) if part else None,mrr=float(np.mean([r['metrics'][name]['mrr'] for r in part])) if part else None,group_means={g:float(np.mean(v)) for g,v in groups.items()})
                summary[role][pool][control]=methods
    gate={};train=[r for r in result if r['role']=='train' and r['pool']=='energy_matched' and r['control']=='actual']
    for baseline in ('Duration','Energy','Global'):
        groups=collections.defaultdict(list)
        for r in train:groups[r['source_group']].append(r['metrics']['Diagonal']['recall_at1']-r['metrics'][baseline]['recall_at1'])
        values=[float(np.mean(v)) for v in groups.values()];ci=group_ci(values,cfg['seed'],cfg['bootstrap_draws']);gate[baseline]=dict(groups=len(values),mean_difference=float(np.mean(values)) if values else None,cluster_ci95=ci,passed=ci[0] is not None and ci[0]>0)
    zeros={r['task']:r for r in result if r['control']=='both_zero'};differences=[r['metrics']['Diagonal']['recall_at1']-zeros[r['task']]['metrics']['Diagonal']['recall_at1'] for r in train];zero_pass=bool(differences) and float(np.mean(differences))>0
    admitted=all(v['passed'] for v in gate.values()) and zero_pass
    np.savez_compressed(out/'arrays.npz',**arrays);value=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),summary=summary,records=result,methods=list(METHODS),train_hybrid_calibration=dict(mean=means.tolist(),std=scales.tolist(),weights=[.5,.5]),p2_gate=dict(train_only=True,comparisons=gate,actual_beats_both_zero=zero_pass,admitted=admitted),benchmark_sha256=sha(Path(benchmark)/'manifest.json'),feature_arrays_sha256=sha(Path(features)/'arrays.npz'),checkpoint_sha256=fm['checkpoint_sha256'],control_population_identical=True,optimizer_steps=0,human_reviewed=False,semantic_event_truth='UNKNOWN',can_route=False,operational_qualified=False);write(out/'result.json',value)
    if replay:
        exact_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');first=read(Path(replay)/'result.json');first.pop('code_commit');value.pop('code_commit');assert first==value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.benchmark,a.features,a.out,a.replay_from)

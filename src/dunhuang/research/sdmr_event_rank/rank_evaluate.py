"""All fixed P2 arms, all registered queries; no winner selection on exposed Val."""
import argparse,collections
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.source_context_r42.contracts import exact_arrays
from .scorer import masked_diagonal,retrieval_metrics

def run(benchmark,features,out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh P2 evaluation')
    out.mkdir(parents=True);meta=read(Path(benchmark)/'manifest.json');fm=read(Path(features)/'result.json');assert fm['benchmark_sha256']==sha(Path(benchmark)/'manifest.json');z=np.load(Path(features)/'arrays.npz');result=[];arrays={}
    cache={}
    def token(i,zero=False):
        key=('z' if zero else 'a')+str(i)
        if key not in cache:cache[key]={n:z[key+'_'+n] for n in ('mg','dg','mt','dt','mv','dv','positions')}
        return cache[key]
    for control in ('actual','both_zero','music_swap','motion_swap'):
        for k,t in enumerate(meta['tasks']):
            qi=t['query'];q=token(fm['donors'][str(qi)] if control=='music_swap' else qi,control=='both_zero');values=[]
            for i in t['candidates']:
                c=token(fm['donors'][str(i)] if control=='motion_swap' else i,control=='both_zero');diagonal,_=masked_diagonal(q['mt'],c['dt'],q['mv'],c['dv'],q['positions'],stride=meta['config']['token_stride']);values.append([float(q['mg']@c['dg']),diagonal])
            values=np.array(values);arrays[f'{control}_{k}']=values;result.append(dict(task=k,role=t['role'],pool=t['pool'],group=t['source_group'],control=control,metrics={name:retrieval_metrics(values[:,j],t['positive']) for j,name in enumerate(('Global','Diagonal'))}))
    summary={}
    for role in ('train','val'):
        for pool in ('same_music','energy_matched','cross_music'):
            for control in ('actual','both_zero','music_swap','motion_swap'):
                part=[r for r in result if r['role']==role and r['pool']==pool and r['control']==control];methods={}
                for method in ('Global','Diagonal'):
                    groups=collections.defaultdict(list)
                    for r in part:groups[r['group']].append(r['metrics'][method]['recall_at1'])
                    methods[method]=dict(queries=len(part),music_groups=len(groups),group_macro_recall_at1=float(np.mean([np.mean(v) for v in groups.values()])),group_means={g:float(np.mean(v)) for g,v in groups.items()},recall_at5=float(np.mean([r['metrics'][method]['recall_at5'] for r in part])),mrr=float(np.mean([r['metrics'][method]['mrr'] for r in part])))
                summary['/'.join((role,pool,control))]=methods
    np.savez_compressed(out/'arrays.npz',**arrays);value=dict(summary=summary,records=result,benchmark_sha256=sha(Path(benchmark)/'manifest.json'),feature_arrays_sha256=sha(Path(features)/'arrays.npz'),checkpoint_sha256=fm['checkpoint_sha256'],optimizer_steps=0,model_selection=False,can_route=False,operational_qualified=False);write(out/'result.json',value);print('E2_P2_EVALUATED',len(result),flush=True)
    if replay:exact_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert value==read(Path(replay)/'result.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.benchmark,a.features,a.out,a.replay_from)

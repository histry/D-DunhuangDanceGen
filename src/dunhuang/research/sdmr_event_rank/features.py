"""The frozen R41 checkpoint, its Train normalizer and real feature support are bound."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.coverage_r34.common import forbid_optimizers
from dunhuang.research.paired_event_r41.population import Population
from dunhuang.research.paired_event_r41.model import EventContent
from dunhuang.research.paired_event_r41.train import seeded,digest
from dunhuang.research.source_context_r42.contracts import exact_arrays
from .scorer import R41EventScorer


def load_model(cfg):
    seeded(cfg['seed']);parent=Path('outputs/paired_event_r41_01');expected=next(r['sha256'] for r in read(parent/'artifacts_manifest.json')['files'] if r['path']=='main/model.pt');assert sha(cfg['checkpoint'])==expected
    model=EventContent().cuda().eval();model.load_state_dict(torch.load(cfg['checkpoint'],map_location='cuda',weights_only=True)['state']);model.requires_grad_(False)
    if sum(p.numel() for p in model.parameters())!=522632:raise ValueError('Frozen R41 architecture identity')
    return model


def inputs(pop,rows,ids,zero=False):
    M=[];D=[];DM=[];MV=[]
    for i in ids:
        r=rows[i];z=pop.unit(r['parent_index']);a,b=r['slice'];m=z['music'][a:b].copy();d=z['motion'][a:b].copy();dm=z['motion_mask'][a:b].copy();mv=z['music_mask'][a:b].any(-1)
        if len(m)!=r['frames'] or not mv.any():raise ValueError('Real unclipped common-clock support')
        if zero:m[:]=0;d[:]=0
        M.append(m);D.append(d);DM.append(dm);MV.append(mv)
    return [torch.tensor(np.stack(v),device='cuda') for v in (M,D,DM,MV)]


def build(benchmark,out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh frozen features')
    out.mkdir(parents=True);meta=read(Path(benchmark)/'manifest.json');cfg=meta['config'];rows=meta['rows'];pop=Population(cfg['data']);forbid_optimizers();model=load_model(cfg);before=digest(model)
    service=R41EventScorer(model,cfg,meta['normalizer']);used={t['query'] for t in meta['tasks']}|{i for t in meta['tasks'] for i in t['candidates']};donors={}
    bylength={}
    for i,r in enumerate(rows):bylength.setdefault((r['role'],r['frames']),[]).append(i)
    for i in sorted(used):
        opts=[j for j in bylength[rows[i]['role'],rows[i]['frames']] if rows[j]['audio_sha256']!=rows[i]['audio_sha256']]
        donors[i]=min(opts,key=lambda j:rows[j]['uid']) if opts else None
    for task in meta['tasks']:
        if any(donors[i] is None for i in [task['query']]+task['candidates']):raise ValueError('Counterfactual donor unavailable; repair pre-score population, never fake data')
    used|={v for v in donors.values() if v is not None};arrays={}
    with torch.no_grad():
        for length in cfg['lengths']:
            ids=sorted(i for i in used if rows[i]['frames']==length)
            for start in range(0,len(ids),16):
                batch=ids[start:start+16]
                for zero in (False,True):
                    data=inputs(pop,rows,batch,zero);mg,dg,mt,dt,mv,dv=service.encode_normalized(data);prefix='z' if zero else 'a'
                    for j,i in enumerate(batch):
                        for name,v in [('mg',mg),('dg',dg),('mt',mt),('dt',dt),('mv',mv),('dv',dv)]:arrays[f'{prefix}{i}_{name}']=v[j].cpu().numpy()
                        arrays[f'{prefix}{i}_positions']=np.arange(6,length-6,dtype=np.int32)
                print('E2_FEATURES',length,start+len(batch),len(ids),flush=True)
    assert digest(model)==before;np.savez_compressed(out/'arrays.npz',**arrays);result=dict(benchmark_sha256=sha(Path(benchmark)/'manifest.json'),checkpoint_sha256=sha(cfg['checkpoint']),normalizer=meta['normalizer'],model_digest=before,model_parameters=522632,used_indices=sorted(used),donors={str(k):v for k,v in donors.items()},optimizer_steps=0,test_tensor_reads=0,can_route=False,operational_qualified=False);write(out/'result.json',result)
    if replay:exact_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'result.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();build(a.benchmark,a.out,a.replay_from)

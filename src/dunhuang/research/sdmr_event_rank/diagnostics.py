"""Fixed token-clock retention and all six shared-branch gradient tradeoffs."""
import argparse,itertools
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.contract_r40.crops import tie_accuracy
from dunhuang.research.source_context_r42.contracts import exact_arrays

def run(root,out,replay=None):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh fixed diagnostics')
    out.mkdir(parents=True);meta=read(root/'benchmark/manifest.json');arrays={};report={};offsets=np.array([-15,-8,0,8,15]);q=np.arange(30,60)-6;lineage={}
    for arm in ('Frozen','NoRank','Rank','ShuffledRank'):
        feature=root/('features' if arm=='Frozen' else 'features_'+arm);z=np.load(feature/'arrays.npz');lineage[arm]=dict(result_sha256=sha(feature/'result.json'),arrays_sha256=sha(feature/'arrays.npz'));records=[]
        for i in meta['queries']:
            if f'a{i}_mt' not in z.files:continue
            r=meta['rows'][i];values={}
            for control,prefix in [('actual','a'),('both_zero','z')]:
                m=z[f'{prefix}{i}_mt'];d=z[f'{prefix}{i}_dt'];mv=z[f'{prefix}{i}_mv'];dv=z[f'{prefix}{i}_dv'];v=np.zeros((5,5))
                for a,shift in enumerate(offsets):
                    for b,k in enumerate(offsets):
                        mask=mv[q+shift]&dv[q+k]
                        if not mask.any():raise ValueError('No genuine common clock support')
                        v[a,b]=(m[q+shift][mask]*d[q+k][mask]).sum(-1).mean()
                values[control]=v;arrays[f'{arm}_{i}_{control}']=v
            records.append(dict(query=i,role=r['role'],group=r['source_group'],actual=float(tie_accuracy(values['actual'],np.arange(5)).mean()),both_zero=float(tie_accuracy(values['both_zero'],np.arange(5)).mean())))
        summary={}
        for role in ('train','val'):
            part=[r for r in records if r['role']==role];summary[role]={}
            for control in ('actual','both_zero'):
                groups={g:float(np.mean([r[control] for r in part if r['group']==g])) for g in sorted({r['group'] for r in part})};summary[role][control]=dict(queries=len(part),groups=groups,group_macro_accuracy=float(np.mean(list(groups.values()))))
        gradients=[]
        if arm!='Frozen':
            for step in (0,120,239):
                g=np.load(root/('p2_'+arm)/f'gradient{step}.npz');norms={k:float(np.linalg.norm(g[k])) for k in g.files};cos={a+'__'+b:float(np.dot(g[a],g[b])/max(norms[a]*norms[b],1e-20)) for a,b in itertools.combinations(g.files,2)};gradients.append(dict(step=step,norms=norms,cosines=cos,effective_rank_weight=0 if arm=='NoRank' else 1))
        report[arm]=dict(clock=summary,records=records,all_six_shared_input_gradient_objectives=gradients)
    np.savez_compressed(out/'arrays.npz',**arrays);result=dict(models=report,lineage=lineage,task='CONTROLLED_TOKEN_QUERY_CLOCK_SHIFTS_WITHIN_REAL_PAIRED_SUBWINDOWS',offsets=offsets.tolist(),truth_scope='KNOWN_INJECTED_TOKEN_CLOCK_NOT_SEMANTIC_EVENT_CORRESPONDENCE',same_frozen_queries=True,model_selection=False,optimizer_steps=0,can_route=False);write(out/'result.json',result)
    if replay:exact_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'result.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.root,a.out,a.replay_from)

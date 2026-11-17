"""Actual 30Hz Root/world proxies and official FK replay; not contact-grounded physics."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.evaluate import metrics
from .archive import Arrays
from .motion import decode,world

def run(main,out,official=False):
    root=Path(main);meta=read(root/'result.json');z=Arrays(root/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];rt=None;records=[];count=0;maxerror=0.
    if official:
        import torch
        from dunhuang.research.bridge_r38.study import runtime
        from dunhuang.research.coverage_r34.common import forbid_optimizers
        forbid_optimizers();rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'))
    try:
        for r in meta['records']:
            key=r['key'];values={}
            for name in r['methods']:
                rows=[]
                for seed,x in enumerate(z[key+'_'+name+'_139']):
                    R,root=decode(x);actual=world(x,J,parents);assert np.max(abs(actual-z[key+'_'+name+'_22'][seed]))<2e-6
                    if rt:
                        with torch.no_grad():f=rt.f(torch.tensor(x[None],device='cuda'),rt.s)[0,:,:22].cpu().numpy()
                        error=float(abs(actual-f).max());assert error<2e-6;maxerror=max(maxerror,error);count+=1
                    v=metrics(R,z[key+'_reference_R'],root,J,parents,boundaries=(8,99,104,152,158,248),gap=(104,152));v['source_error_scope']='ORIGINAL_CLOCK_ERROR_INCLUDES_INTENTIONAL_RETIMING; NOT_SOURCE_PATH_LOSS'
                    for n in (1,2,3):v['model_root_D'+str(n)]=float(np.sqrt(np.mean((np.diff(root,n=n,axis=0)*30**n)**2)));v['world_fk_D'+str(n)]=float(np.sqrt(np.mean((np.diff(actual,n=n,axis=0)*30**n)**2)))
                    rows.append(v)
                values[name]=rows
            records.append(dict(key=key,methods=values,hidden_truth_is_continuous=r['source']['domain']!='ChangE-cross',cross_source_gap_error_scope='NO_CONTINUOUS_TRUTH_FOR_CHANGE_CROSS',world_root_truth='UNKNOWN_FOR_CHANGE_MODEL_PRIOR_ONLY',contact_truth='UNKNOWN',full_native24='NOT_EVALUABLE_22_OF24'))
        if rt:rt.check()
    finally:
        if rt:rt.close()
        z.close()
    write(out,dict(records=records,arrays_sha256=sha(Path(main)/'arrays.npz'),official_fk_executed=official,official_fk_trajectories=count,maximum_fk_error_m=maxerror,optimizer_calls=0,new_neural_samples=0,full_native24='NOT_EVALUABLE_22_OF24',foot_slide='NOT_EVALUABLE_NO_CONTACT_TRUTH',operational_qualified=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);p.add_argument('--official',action='store_true');a=p.parse_args();run(a.main,a.out,a.official)

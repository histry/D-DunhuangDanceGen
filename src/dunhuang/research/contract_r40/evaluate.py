"""Saved-output CPU verification and scope-separated real-target errors."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.transition_r37.evaluate import decode,world_fk,metrics,curves
from dunhuang.research.stabilize_r39.evaluate import root_profile
from dunhuang.research.transition_r37.geometry import c2_envelope


def run(root,out,official=False):
    root=Path(root);z=np.load(root/'generation/arrays.npz');meta=read(root/'generation/result.json');J=z['skeleton_J22'];parents=z['parents'];records=[];checks=[];rt=None
    if official:
        import torch
        from dunhuang.research.bridge_r38.study import runtime
        from dunhuang.research.coverage_r34.common import forbid_optimizers
        rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'));forbid_optimizers()
    for row in meta['records']:
        key=row['key'];lo,hi=row['gap'];ref=z[key+'_reference_R'];Q=relative_fk(ref,J,parents);base=z[key+'_Geometry_139'];methods={};diags=[{},{}]
        for arm in ('Geometry','Frozen','Matched','ResidualRTS'):
            scores=[]
            for seed,raw in enumerate(z[key+'_'+arm+'_139']):
                if not np.isfinite(raw).all():raise ValueError('Nonfinite output')
                R,rootvalue=decode(raw);P=relative_fk(R,J,parents);world=world_fk(R,rootvalue,J,parents)
                assert abs(world-z[key+'_'+arm+'_22'][seed]).max()<2e-6
                if rt is not None:
                    with torch.no_grad():actual=rt.f(torch.tensor(raw[None],device='cuda'),rt.s)[0,:,:22].cpu().numpy()
                    if abs(actual-world).max()>2e-6:raise ValueError('Official Lodge FK differs from rendered states')
                assert np.array_equal(raw[:lo],base[seed,:lo]) and np.array_equal(raw[hi:],base[seed,hi:])
                B,_=decode(base[seed]);angle=Rotation.from_matrix((B.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(256,22);w=c2_envelope(256,row['gap'])
                assert (angle<=.6*w[:,None]+2e-6).all();assert (np.linalg.norm(rootvalue-base[seed,:,4:7],axis=-1)<=.5*w+2e-6).all()
                result=metrics(R,ref,rootvalue,J,parents,row['cuts'],row['gap'],128);result['gap_fk_rmse_m']=float(np.sqrt(np.mean((P[lo:hi]-Q[lo:hi])**2)));result['root']=root_profile(rootvalue,row['cuts']);result['body_gap_amplitude_rms_m']=float(np.sqrt(np.mean(np.ptp(P[lo:hi],axis=0)**2)))
                result['regions']={}
                for name,idx in [('gap',slice(lo,hi)),('left_boundary',slice(lo-3,lo+4)),('right_boundary',slice(hi-4,hi+3)),('whole',slice(None))]:
                    e=(P-Q)[idx];d={f'body_D{k}':float(np.sqrt(np.mean((np.diff(e,n=k,axis=0)*30**k)**2))) for k in range(4)}
                    if row['reference_is_target']:
                        er=(rootvalue-z[key+'_target_139'][seed,:,4:7])[idx];d.update({f'root_D{k}':float(np.sqrt(np.mean((np.diff(er,n=k,axis=0)*30**k)**2))) for k in range(4)})
                    result['regions'][name]=d
                scores.append(result);v=curves(R,ref,rootvalue,J,parents);v['root_acceleration']=result['root']['acceleration_curve'];v['root_jerk']=result['root']['jerk_curve'];diags[seed][arm]=v;checks.append(dict(case=key,arm=arm,seed=seed,protected_exact=True,cpu_fk=True,budgets=True))
            methods[arm]=scores
        records.append(dict(key=key,methods=methods,diagnostics_variants=diags,reference_is_target=row['reference_is_target'],r39=True))
    if rt is not None:rt.check();rt.close()
    result=dict(records=records,verification=checks,official_fk_executed=official,arrays_sha256=sha(root/'generation/arrays.npz'),optimizer_steps=0,whole_song_generations=0,operational_qualified=False,full_native='NOT_EVALUABLE_22_OF24',foot_contact_truth='UNKNOWN');write(out,result);print('R40_CPU_VERIFY',len(checks),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--official',action='store_true');a=p.parse_args();run(a.root,a.out,a.official)

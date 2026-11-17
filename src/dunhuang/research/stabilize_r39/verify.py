"""Independent budget/endpoint audit; optional parity with the actual official Lodge FK."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.evaluate import decode,world_fk
from dunhuang.research.transition_r37.geometry import c2_envelope

def run(root,out,official=False):
    root=Path(root);a=np.load(root/'generation/arrays.npz');old=np.load('outputs/bridge_r38_01/evaluation/arrays.npz');meta=read(root/'generation/result.json');metrics=read(root/'evaluation.json');cal=read(root/'calibration/result.json')['parameters'];stats=np.load(root/'calibration/arrays.npz');J=a['skeleton_J22'];parents=a['parents'];rows=[];rt=None
    if official:
        import torch
        from dunhuang.research.bridge_r38.study import runtime
        from dunhuang.research.coverage_r34.common import forbid_optimizers
        rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'));forbid_optimizers()
    for row in meta['records']:
        key=row['key'];lo,hi=row['gap'];w=c2_envelope(256,row['gap']);anchor=a[key+'_Geometry_139'];reference=old[key+'_Trained_139']
        if not np.array_equal(a[key+'_Raw_139'],reference):raise ValueError('R38 baseline no longer exact')
        for mode in ('Geometry','Raw','RootContinuous','ResidualRTS'):
            for i,raw in enumerate(a[key+'_'+mode+'_139']):
                if not np.isfinite(raw).all():raise ValueError('Nonfinite final motion')
                R,Root=decode(raw);P=world_fk(R,Root,J,parents);saved=a[key+'_'+mode+'_22'][i]
                cpu_error=float(abs(P-saved).max());official_error=None
                if rt is not None:
                    with torch.no_grad():actual=rt.f(torch.tensor(raw[None],device='cuda'),rt.s)[0,:,:22].cpu().numpy()
                    official_error=float(abs(actual-saved).max())
                    if official_error>2e-6:raise ValueError(('Official Lodge FK mismatch',key,mode,official_error))
                if cpu_error>2e-6:raise ValueError('CPU FK mismatch')
                q=np.linalg.norm(np.diff(Root.astype(float),n=2,axis=0)*900,axis=-1);registered=next(r for r in metrics['records'] if r['key']==key)['methods'][mode][i]['root']['acceleration_peak']
                if float(q.max())!=registered:raise ValueError('Root derivative readout differs')
                if mode in ('RootContinuous','ResidualRTS'):
                    delta=Root-anchor[i,:,4:7];B,_=decode(anchor[i]);angle=Rotation.from_matrix((B.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(256,22)
                    if (np.linalg.norm(delta,axis=-1)>cal['root_budget_m']*w+2e-6).any() or (angle>.6*w[:,None]+2e-6).any():raise ValueError('Physical-meter/radian budget violation')
                    if not np.array_equal(raw[:lo],anchor[i,:lo]) or not np.array_equal(raw[hi:],anchor[i,hi:]):raise ValueError('Read-only source modified')
                if mode=='ResidualRTS':
                    continuous=a[key+'_RootContinuous_139'][i]
                    if not np.array_equal(raw[:,:4],continuous[:,:4]):raise ValueError('RTS changed C4')
                    if not np.array_equal(raw[lo:lo+3,4:],anchor[i,lo:lo+3,4:]) or not np.array_equal(raw[hi-3:hi,4:],anchor[i,hi-3:hi,4:]):raise ValueError('Solved boundary equality lost')
                rows.append(dict(case=key,arm=mode,seed=row['seeds'][i],cpu_fk_max_error_m=cpu_error,official_fk_max_error_m=official_error,root_peak_acceleration_m_s2=float(q.max())))
    if rt is not None:rt.check();rt.close()
    write(out,dict(records=rows,trajectories=len(rows),official_fk_executed=official,optimizer_steps=0,arrays_sha256=sha(root/'generation/arrays.npz'),native24='NOT_EVALUABLE_22_OF24',contact_truth='UNKNOWN'))
    print('R39_INDEPENDENT_VERIFY',len(rows),'official',official,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--official',action='store_true');a=p.parse_args();run(a.root,a.out,a.official)

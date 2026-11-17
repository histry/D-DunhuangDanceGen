"""Separate absolute kinematics from paired target errors; saved outputs only."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,relative_fk
from dunhuang.research.transition_r37.evaluate import decode
from dunhuang.research.coverage_r34.common import forbid_optimizers
from .post import frozen_rts


def run(root,out,replay=None):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh absolute motion readout')
    forbid_optimizers();result=[]
    for folder in ('main_generation','real_generation','aist_generation'):
     p=root/folder
     if not (p/'result.json').exists():raise FileNotFoundError(p/'result.json')
     z=np.load(p/'arrays.npz');meta=read(p/'result.json');J=z['skeleton_J22'];parents=z['parents']
     for row in meta['records']:
      key=row['key'];Q=relative_fk(z[key+'_reference_R'],J,parents);methods={}
      for name in row['methods']+['FrozenRTS']:
       xs=frozen_rts(z[key+'_Frozen_139'],z[key+'_Geometry_139']) if name=='FrozenRTS' else z[key+'_'+name+'_139'];values=[]
       for x in xs:
        R,_=decode(x);P=relative_fk(R,J,parents);values.append(dict(activity_rms_mps=float(np.sqrt(np.mean((np.diff(P[104:152],axis=0)*30)**2))),absolute_body_jerk_rms_mps3=float(np.sqrt(np.mean((np.diff(P[101:155],n=3,axis=0)*27000)**2)))))
       methods[name]=values
      result.append(dict(folder=folder,key=key,role=row['role'],recording_uid=row['recording_uid'],domain=row['domain'],source_activity_rms_mps=float(np.sqrt(np.mean((np.diff(Q[104:152],axis=0)*30)**2))),source_jerk_rms_mps3=float(np.sqrt(np.mean((np.diff(Q[101:155],n=3,axis=0)*27000)**2))),methods=methods))
    identity=[]
    meta=read('outputs/transition_r37_02/generation/result.json')
    for r in meta['records']:identity.append(dict(key=r['key'],song=r.get('song'),window_start=r['binding']['window_start'],window_frames=256))
    value=dict(records=result,music=identity,scope='ABSOLUTE_RMS_KINEMATIC_PROXIES_NOT_NATIVE_PASS_NOT_CONTACT_TRUTH',optimizer_steps=0)
    if replay:assert value==read(replay)
    write(out,value)
    print('R42_ABSOLUTE_READOUT',len(result),identity,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.root,a.out,a.replay_from)

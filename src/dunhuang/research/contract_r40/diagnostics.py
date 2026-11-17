"""Bounded-target reference and saturation diagnostics, never a feasibility proof."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,relative_fk
from dunhuang.research.transition_r37.evaluate import decode
from dunhuang.research.transition_r37.geometry import c2_envelope


def run(root,out):
    root=Path(root);z=np.load(root/'generation/arrays.npz');meta=read(root/'generation/result.json');J=z['skeleton_J22'];parents=z['parents'];s=np.load('outputs/stabilize_r39_01/calibration/arrays.npz');scale=s['root_normalizer_scale'];offset=s['root_normalizer_offset'];records=[]
    for row in meta['records']:
        key=row['key'];lo,hi=row['gap'];w=c2_envelope(256,row['gap']);live=w>0;base=z[key+'_Geometry_139'];trace=z[key+'_root_trace'];proposal=trace[:,:,0]*scale+offset;saturated=float((abs(proposal)>=1-1e-6).mean());modes={}
        for name in ('Frozen','Matched','ResidualRTS'):
            fractions=[]
            for a,p in zip(base,z[key+'_'+name+'_139']):
                B,_=decode(a);P,_=decode(p);angle=Rotation.from_matrix((B.swapaxes(-1,-2)@P).reshape(-1,3,3)).magnitude().reshape(256,22);fractions.append(float((angle[live]>=.6*w[live,None]-1e-5).mean()))
            modes[name]=dict(body_at_budget_fraction=fractions)
        target_reference=None
        if row['reference_is_target']:
            a=base[0];truth=z[key+'_target_139'][0];B,_=decode(a);T,_=decode(truth);v=Rotation.from_matrix((B.swapaxes(-1,-2)@T).reshape(-1,3,3)).as_rotvec().reshape(256,22,3);angle=np.linalg.norm(v,axis=-1);gain=np.minimum(1,.6*w[:,None]/np.maximum(angle,1e-12));P=B@Rotation.from_rotvec((v*gain[...,None]).reshape(-1,3)).as_matrix().reshape(256,22,3,3);points=relative_fk(P,J,parents);q=relative_fk(T,J,parents)
            target_reference=dict(true_target_outside_body_budget_fraction=float((angle[live]>.6*w[live,None]+1e-6).mean()),clipped_target_gap_fk_rmse_m=float(np.sqrt(np.mean((points[lo:hi]-q[lo:hi])**2))),scope='PER_JOINT_TARGET_ROTATION_PROJECTION; NOT_CARTESIAN_OPTIMUM_OR_DYNAMICS_FEASIBILITY_BOUND')
        records.append(dict(key=key,modes=modes,actual_cfg_root_proposal_at_normalizer_box_fraction=saturated,target_reference=target_reference,source_residual_gate_at_center=float(1-w[128]),source_gate_causality='NOT_TESTED; SELF_ATTENTION_CAN_STILL_PROPAGATE_VISIBLE_SOURCE_INFORMATION'))
    write(out,dict(records=records,optimizer_steps=0,selection_or_training_changed=False));print('R40_PROJECTION_DIAGNOSTICS',len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)

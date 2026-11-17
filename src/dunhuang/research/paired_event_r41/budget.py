"""Train-only target-informed edit-space diagnostic, never a generator or inference path."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.transition_r37.evaluate import decode
from dunhuang.research.transition_r37.geometry import c2_envelope

CHAINS={'pelvis_trunk':[0,3,6,9,12,15],'left_shoulder_wrist':[0,3,6,9,13,16,18,20],'right_shoulder_wrist':[0,3,6,9,14,17,19,21],'left_leg':[0,1,4,7,10],'right_leg':[0,2,5,8,11]}
STEPS=80


def summarize(raw,truth,geometry,J,parents):
    w=c2_envelope(256,[104,152]);regions=dict(zero_budget=(np.arange(256)>=104)&(np.arange(256)<152)&(w==0),ramp=(w>0)&(w<1),central=w==1);records=[]
    for x,t,g in zip(raw,truth,geometry):
        R,root=decode(x);T,tr=decode(t);B,_=decode(g);P=relative_fk(R,J,parents);Q=relative_fk(T,J,parents);theta=Rotation.from_matrix((B.swapaxes(-1,-2)@T).reshape(-1,3,3)).magnitude().reshape(256,22);row={}
        for name,mask in regions.items():
            row[name]=dict(frames=int(mask.sum()),outside_fraction=float((theta[mask]>.6*w[mask,None]+1e-6).mean()),rotation_error_lower_bound_rad=float(np.maximum(0,theta[mask]-.6*w[mask,None]).mean()),fk_rmse_m=float(np.sqrt(np.mean((P[mask]-Q[mask])**2))))
        e=(P-Q)[101:155];er=(root-tr)[101:155];row['boundary_supported_errors']={f'body_D{k}':float(np.sqrt(np.mean((np.diff(e,n=k,axis=0)*30**k)**2))) for k in range(4)};row['boundary_supported_errors'].update({f'root_D{k}':float(np.sqrt(np.mean((np.diff(er,n=k,axis=0)*30**k)**2))) for k in range(3)})
        row['chains']={name:dict(ancestor_budget_conflict_fraction=float((theta[104:152,ids]>.6*w[104:152,None]+1e-6).mean()),endpoint_fk_error_m=float(np.sqrt(np.mean((P[104:152,ids[-1]]-Q[104:152,ids[-1]])**2)))) for name,ids in CHAINS.items()};records.append(row)
    return records


def run(out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh diagnostic output')
    out.mkdir(parents=True);path=Path('outputs/stabilize_r39_01/calibration/arrays.npz');z=np.load(path);truth=z['truth139'];geometry=z['geometry139'];J=z['skeleton_J22'];parents=z['parents'];meta=read('outputs/stabilize_r39_01/calibration/result.json');cases=meta['cases'];arrays=dict(truth139=truth,geometry139=geometry,skeleton_J22=J,parents=parents);history={}
    if replay:
        old=np.load(Path(replay)/'arrays.npz')
        for name in ('sampler_box','rts_equalities'):arrays[name+'_139']=old[name+'_139']
    else:
        import torch
        from pytorch3d.transforms import axis_angle_to_matrix
        from .train import seeded
        seeded(41041);G=torch.tensor(np.stack([decode(x)[0] for x in geometry]),device='cuda',dtype=torch.float32);T=torch.tensor(np.stack([relative_fk(decode(t)[0],J,parents) for t in truth]),device='cuda',dtype=torch.float32);base=torch.tensor(geometry[:,:,4:7],device='cuda');target_root=torch.tensor(truth[:,:,4:7],device='cuda');jt=torch.tensor(J,device='cuda');scale=torch.tensor(z['root_normalizer_scale'],device='cuda');offset=torch.tensor(z['root_normalizer_offset'],device='cuda');lower=(-1-offset)/scale;upper=(1-offset)/scale
        def fk(R):
            p=[torch.zeros_like(R[:,:,0,:,0])];world=[R[:,:,0]]
            for j in range(1,22):k=int(parents[j]);p.append(p[k]+(world[k]@(jt[j]-jt[k])[None,None,:,None])[...,0]);world.append(world[k]@R[:,:,j])
            return torch.stack(p,2)
        initial=fk(G)-T;root_initial=base-target_root;body_scales=[torch.diff(initial[:,101:155],n=k,dim=1).mul(30**k).square().mean().sqrt().clamp_min(.01) for k in range(4)];root_scales=[torch.diff(root_initial[:,101:155],n=k,dim=1).mul(30**k).square().mean().sqrt().clamp_min(.01) for k in range(3)]
        for mode in ('sampler_box','rts_equalities'):
            w=c2_envelope(256,[104,152]).copy()
            if mode=='rts_equalities':w[104:107]=0;w[149:152]=0
            w=torch.tensor(w,device='cuda',dtype=torch.float32);rv=torch.nn.Parameter(torch.zeros(len(G),256,22,3,device='cuda'));root=torch.nn.Parameter(torch.zeros_like(base));opt=torch.optim.Adam([rv,root],lr=.02);rows=[]
            for step in range(STEPS):
                R=G@axis_angle_to_matrix(rv);e=(fk(R)-T)[:,101:155];er=(base+root-target_root)[:,101:155];value=sum(torch.nn.functional.smooth_l1_loss(torch.diff(e,n=k,dim=1)*30**k/body_scales[k],torch.zeros_like(e[:,k:])) for k in range(4))+sum(torch.nn.functional.smooth_l1_loss(torch.diff(er,n=k,dim=1)*30**k/root_scales[k],torch.zeros_like(er[:,k:])) for k in range(3));opt.zero_grad();value.backward();opt.step()
                with torch.no_grad():
                    rv.mul_(torch.minimum(torch.ones_like(rv[...,0]),.6*w[None,:,None]/rv.norm(dim=-1).clamp_min(1e-8))[...,None]);root.mul_(torch.minimum(torch.ones_like(root[...,0]),.5*w[None]/root.norm(dim=-1).clamp_min(1e-8))[...,None]);root.copy_((base+root).clamp(lower,upper)-base);rv[:,w==0]=0;root[:,w==0]=0
                rows.append(dict(step=step,objective=float(value.detach())))
            with torch.no_grad():R=(G@axis_angle_to_matrix(rv)).cpu().numpy();rr=(base+root).cpu().numpy()
            raw=geometry.copy();raw[:,104:152,7:]=R[:,104:152,:,:2,:].reshape(len(G),48,132);raw[:,104:152,4:7]=rr[:,104:152];locked=w.cpu().numpy()==0;raw[:,locked]=geometry[:,locked];arrays[mode+'_139']=raw;history[mode]=rows
    # Pointwise target projection is a comparison, not a global FK optimum.
    projected=geometry.copy();w=c2_envelope(256,[104,152])
    for i,(t,g) in enumerate(zip(truth,geometry)):
        R,_=decode(t);B,_=decode(g);v=Rotation.from_matrix((B.swapaxes(-1,-2)@R).reshape(-1,3,3)).as_rotvec().reshape(256,22,3);a=np.linalg.norm(v,axis=-1);v*=np.minimum(1,.6*w[:,None]/np.maximum(a,1e-12))[...,None];P=B@Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(256,22,3,3);projected[i,104:152,7:]=P[104:152,:,:2,:].reshape(48,132)
    arrays['target_point_projection_139']=projected
    for mode in ('sampler_box','rts_equalities'):
        raw=arrays[mode+'_139'];budget=c2_envelope(256,[104,152]).copy()
        if mode=='rts_equalities':budget[104:107]=0;budget[149:152]=0
        for x,g in zip(raw,geometry):
            R,root=decode(x);B,base=decode(g);a=Rotation.from_matrix((B.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(256,22);assert (a<=.6*budget[:,None]+5e-6).all();assert (np.linalg.norm(root-base,axis=-1)<=.5*budget+5e-6).all();assert np.array_equal(x[budget==0],g[budget==0])
    np.savez_compressed(out/'arrays.npz',**arrays);result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_arrays_sha256=sha(path),cases=cases,roles='FROZEN_R39_TRAIN_ONLY_NO_VAL',metrics={name:summarize(arrays[key],truth,geometry,J,parents) for name,key in [('Geometry','geometry139'),('PointwiseTargetProjection','target_point_projection_139'),('FeasibleSamplerBox','sampler_box_139'),('FeasibleRTSEqualities','rts_equalities_139')]},history=history,network_optimizer_steps=0,diagnostic_motion_optimizer_steps=0 if replay else 2*STEPS,scope='TARGET_INFORMED_TRAIN_ONLY_FEASIBLE_SEARCH; NOT_MODEL_NOT_INFERENCE_NOT_GLOBAL_OPTIMUM',chain_scope='KINEMATIC_ANCESTOR_CONFLICT_AND_ENDPOINT_ERROR_ASSOCIATION_NOT_CAUSAL_ATTRIBUTION',whole_song_generations=0,operational_qualified=False);write(out/'result.json',result)
    if replay:
        old=read(Path(replay)/'result.json');assert result['metrics']==old['metrics'];a=np.load(Path(replay)/'arrays.npz')
        for k in arrays:assert np.array_equal(arrays[k],a[k]),k
    print('R41_BUDGET_DIAGNOSTIC',len(cases),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.out,a.replay_from)

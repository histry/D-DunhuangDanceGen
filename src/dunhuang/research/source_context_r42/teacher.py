"""Train-only joint feasible trajectories, never consulted by held inference."""
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix,rotation_6d_to_matrix,matrix_to_axis_angle,matrix_to_axis_angle
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.paired_event_r41.train import seeded


def fk(R,J,parents):
    p=[torch.zeros_like(R[:,:,0,:,0])];world=[R[:,:,0]]
    for j in range(1,22):
        k=int(parents[j]);p.append(p[k]+(world[k]@(J[j]-J[k])[None,None,:,None])[...,0]);world.append(world[k]@R[:,:,j])
    return torch.stack(p,2)


def scales_from_errors(error,root_error,root_valid):
    bs=torch.stack([torch.diff(error[:,101:155],n=k,dim=1).mul(30**k).square().mean().sqrt().clamp_min(.01) for k in range(4)])
    rs=[]
    for k in range(3):
        e=torch.diff(root_error[:,101:155],n=k,dim=1)*30**k
        if k==0:e=e[root_valid]
        rs.append(e.square().mean().sqrt().clamp_min(.01) if e.numel() else root_error.new_tensor(1.))
    return bs,torch.stack(rs)


def temporal_loss(error,root_error,root_valid,bs,rs):
    parts={}
    for k in range(4):
        e=torch.diff(error[:,101:155],n=k,dim=1)*30**k/bs[k];parts['body_D'+str(k)]=torch.nn.functional.smooth_l1_loss(e,torch.zeros_like(e))
    for k in range(3):
        e=torch.diff(root_error[:,101:155],n=k,dim=1)*30**k/rs[k]
        if k==0:e=e[root_valid]
        parts['root_D'+str(k)]=torch.nn.functional.smooth_l1_loss(e,torch.zeros_like(e)) if e.numel() else root_error.sum()*0
    return sum(parts.values()),parts


def run(data,out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh teacher output')
    out.mkdir(parents=True);cfg=read(Path(__file__).with_name('config.json'));data=Path(data);z=np.load(data/'arrays.npz');rows=[r for r in read(data/'result.json')['records'] if r['role']=='train'];assert rows and all(r['role']=='train' for r in rows);seeded(cfg['seed']);device='cuda';keys=[r['key'] for r in rows];T=torch.tensor(np.stack([z[k+'_truth'] for k in keys]),device=device);A=torch.tensor(np.stack([z[k+'_anchor_raw'] for k in keys]),device=device);G=rotation_6d_to_matrix(A[:,:,7:].reshape(len(A),256,22,6));R=rotation_6d_to_matrix(T[:,:,7:].reshape(len(A),256,22,6));J=torch.tensor(z['skeleton_J22'],device=device);parents=z['parents'];Q=fk(R,J,parents);mask=torch.tensor([r['root_position_observed'] for r in rows],device=device);bs,rs=scales_from_errors(fk(G,J,parents)-Q,A[:,:,4:7]-T[:,:,4:7],mask);w=c2_envelope(256,cfg['gap']).copy();w[104:107]=0;w[149:152]=0;w=torch.tensor(w,device=device,dtype=torch.float32);history=[]
    if replay:
        raw=np.load(Path(replay)/'arrays.npz')['teacher139'];updates=0
    else:
        rv=torch.nn.Parameter(torch.zeros(len(A),256,22,3,device=device));dr=torch.nn.Parameter(torch.zeros_like(A[:,:,4:7]));opt=torch.optim.Adam([rv,dr],lr=.02);scale=torch.tensor(z['root_normalizer_scale'],device=device);offset=torch.tensor(z['root_normalizer_offset'],device=device);lower=(-1-offset)/scale;upper=(1-offset)/scale
        for step in range(cfg['teacher_steps']):
            P=fk(G@axis_angle_to_matrix(rv),J,parents);value,parts=temporal_loss(P-Q,A[:,:,4:7]+dr-T[:,:,4:7],mask,bs,rs);opt.zero_grad();value.backward();opt.step()
            with torch.no_grad():
                rv.mul_(torch.minimum(torch.ones_like(rv[...,0]),.6*w[None,:,None]/rv.norm(dim=-1).clamp_min(1e-8))[...,None]);dr.mul_(torch.minimum(torch.ones_like(dr[...,0]),.5*w[None]/dr.norm(dim=-1).clamp_min(1e-8))[...,None]);dr.copy_((A[:,:,4:7]+dr).clamp(lower,upper)-A[:,:,4:7]);rv[:,w==0]=0;dr[:,w==0]=0
            history.append(dict(step=step,loss=float(value.detach()),parts={k:float(v.detach()) for k,v in parts.items()}))
        with torch.no_grad():P=G@axis_angle_to_matrix(rv);raw=A.cpu().numpy().copy();raw[:,:,7:]=P[:,:,:,:2,:].cpu().numpy().reshape(len(A),256,132);raw[:,:,4:7]=(A[:,:,4:7]+dr).cpu().numpy()
        raw[:,w.cpu().numpy()==0]=A.cpu().numpy()[:,w.cpu().numpy()==0];updates=cfg['teacher_steps']
    teacher=torch.tensor(raw,device=device);P=fk(rotation_6d_to_matrix(teacher[:,:,7:].reshape(len(A),256,22,6)),J,parents);value,parts=temporal_loss(P-Q,teacher[:,:,4:7]-T[:,:,4:7],mask,bs,rs);old,oldparts=temporal_loss(fk(G,J,parents)-Q,A[:,:,4:7]-T[:,:,4:7],mask,bs,rs);a=A.cpu().numpy();theta=torch.linalg.norm(matrix_to_axis_angle(G.transpose(-1,-2)@rotation_6d_to_matrix(teacher[:,:,7:].reshape(len(A),256,22,6))),dim=-1);assert (theta<=.6*w[None,:,None]+3e-6).all();assert np.array_equal(raw[:,w.cpu().numpy()==0],a[:,w.cpu().numpy()==0]);assert (torch.linalg.norm(teacher[:,:,4:7]-A[:,:,4:7],dim=-1)<=.5*w+3e-6).all()
    np.savez_compressed(out/'arrays.npz',teacher139=raw,keys=keys,body_scales=bs.cpu().numpy(),root_scales=rs.cpu().numpy());result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_keys=keys,recording_uids=sorted({r['recording_uid'] for r in rows}),train_windows=len(rows),data_sha256=sha(data/'arrays.npz'),network_optimizer_steps=0,motion_optimizer_steps=updates,final_loss=float(value),geometry_loss=float(old),final_parts={k:float(v) for k,v in parts.items()},geometry_parts={k:float(v) for k,v in oldparts.items()},history=history,root_D0_mask=mask.cpu().tolist(),constraints_verified=True,scope='TRAIN_ONLY_TARGET_INFORMED_FEASIBLE_TEACHER_NOT_GENERATOR_NOT_GLOBAL_OPTIMUM',whole_song_generations=0);write(out/'result.json',result)
    if replay:
        from dunhuang.research.coverage_r34.common import same_arrays
        same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');prev=read(Path(replay)/'result.json');assert result['final_parts']==prev['final_parts'] and result['geometry_parts']==prev['geometry_parts']
    print('R42_TEACHER',len(rows),float(old),float(value),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.data,a.out,a.replay_from)

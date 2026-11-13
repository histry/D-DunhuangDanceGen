"""Whole E1-W reference and phase geometry at every denoising step."""
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.contracts import MotionAdapter,Q as NATIVE_Q
from dunhuang.research.source_capacity.common import read
from .reference import build

def prepare(request,normalizer,out,device):
    from pytorch3d.transforms import rotation_6d_to_matrix as matrix,matrix_to_axis_angle as log,axis_angle_to_matrix as exp
    from dunhuang.planning.event_plan import validate
    plan=read(checked(request["event_plan"]));validate(plan,plan["audio_sha256"],plan["frames"])
    cfg=request["reperformance_config"];T=request["frames"];start=request.get("reference_offset",0)
    if not 0<=start<start+T<=plan["frames"]:raise ValueError("Reference crop outside full Event Plan")
    if plan["audio_sha256"]!=request["audio"]["sha256"]:raise ValueError("Reference audio identity differs")
    reference,known,phase,stages=build(plan,cfg)
    group=request["condition_group"];joints=cfg["groups"][group]
    reference=reference[start:start+T];known=known[start:start+T];phase=phase[start:start+T]
    selected=np.zeros(24,bool);selected[joints]=True;known&=selected;phase*=known
    if request.get("edit_interval"):
        a,b=request["edit_interval"];active=np.zeros(T,bool);active[a:b]=True;known&=active[:,None];phase*=known
    edge=MotionAdapter.to_edge(reference)
    target=matrix(torch.tensor(edge[:,7:].reshape(T,24,6),device=device))
    known=torch.tensor(known,device=device);phase=torch.tensor(phase,device=device)
    scale=normalizer.scaler.scale_.to(device);bias=normalizer.scaler.min_.to(device);Q=torch.tensor(NATIVE_Q,device=device)
    ids=torch.arange(T,device=device);windows=torch.clamp(ids//75,max=request["windows"]-1);local=ids-75*windows
    meta=dict(schema="primitive_reperformance_v1",condition_group=group,frames=T,observed_source_joints=list(range(22)),controlled_joints=joints,
        source_root_translation=False,source_contact=False,finger_geometry=False,full_151d_source_truth=False,source_joint_frames=int(known.sum()),
        phase_count=sum(start<=p["music_frame"]<start+T for p in stages),stages=stages,no_reference_clipping=True,no_output_pasting=True,
        root_rotation_policy="source relative turn and fitted tilt; initial heading aligned to generated prediction",
        reference_offset=start,edit_interval=request.get("edit_interval"),semantic_phase_truth="UNKNOWN")
    write(out/"reperformance_reference.json",meta);logs=[];aligned=None
    def guide(predicted,progress):
        nonlocal aligned
        x=predicted[windows,local].clone();raw=(x[:,7:]-bias[7:])/scale[7:];vectors=raw.reshape(T,24,2,3);old=matrix(raw.reshape(T,24,6))
        if aligned is None:
            aligned=target.clone()
            for event in plan["events"]:
                a=max(event["music_interval"][0],start)-start;b=min(event["music_interval"][1],start+T)-start
                if b<=a:continue
                original=Q.T@target[a,0];current=Q.T@old[a,0]
                yaw=lambda r:torch.atan2(r[0,2],r[2,2])
                axis=torch.zeros(3,device=device);axis[1]=yaw(current)-yaw(original)
                aligned[a:b,0]=Q@exp(axis)@Q.T@target[a:b,0]
        omega=log(old.transpose(-1,-2)@aligned)*(phase*progress**2)[...,None]
        result=x[:,7:].reshape(T,24,6).clone();accepted=torch.zeros((T,24),dtype=torch.bool,device=device);factor=torch.ones((T,24),device=device);dose=torch.zeros_like(factor)
        for _ in range(9):
            candidate=old@exp(omega*factor[...,None])
            encoded=(vectors@(old.transpose(-1,-2)@candidate)).reshape(T,24,6)*scale[7:].reshape(24,6)+bias[7:].reshape(24,6)
            good=(encoded.abs().amax(-1)<=1+32*torch.finfo(encoded.dtype).eps)&known&~accepted
            result=torch.where(good[...,None],encoded,result);dose=torch.where(good,factor,dose);accepted|=good;factor*=.5
        x[:,7:]=result.reshape(T,144);output=predicted.clone()
        for w in range(request["windows"]):
            n=min(150,T-75*w)
            if n>0:output[w,:n]=x[75*w:75*w+n]
        logs.append(dict(progress=progress,domain_rejected_joint_frames=int((known&~accepted).sum()),mean_accepted_fraction=float(dose[known].mean()) if known.any() else None))
        if len(logs)==50:write(out/"reperformance_guide.json",dict(config=cfg,steps=logs,reference=meta,neural_weight_updates=0,post_generation_pose_paste=False))
        return output
    return guide

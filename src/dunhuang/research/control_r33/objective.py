"""Paired Root-relative body error on valid target-time templates, never absolute stillness."""
import torch
import torch.nn.functional as F
from dunhuang.research.reference_r26.geometry import fk22
from dunhuang.research.lodge_adapter_r25.model import PARTS
from dunhuang.research.content_r32.partial import partial_loss

def derivative_support(valid,order):
    if valid.ndim!=3 or valid.dtype!=torch.bool or order not in (1,2):raise ValueError("Bool BxTx22 and derivative1/2")
    length=valid.shape[1]-order
    if length<=0:raise ValueError("Too few observed frames")
    mask=valid[:,:length].clone()
    for j in range(1,order+1):mask=mask & valid[:,j:j+length]
    return mask

def paired_error(points,target,valid):
    if points.shape!=target.shape or points.ndim!=4 or points.shape[2:]!=(22,3) or valid.shape!=points.shape[:-1]:raise ValueError("Matched body geometry and validity required")
    if not torch.isfinite(points).all() or not torch.isfinite(target[valid]).all():raise ValueError("Finite observed geometry")
    if not torch.isfinite(target[:,:,0][valid.any(-1)]).all():raise ValueError("A finite relative-coordinate Root gauge is required; no unknown world Root inference")
    relative=points-points[:,:,:1]
    truth=target-target[:,:,:1]
    safe_p=torch.where(valid[...,None],relative,torch.zeros_like(relative))
    safe_t=torch.where(valid[...,None],truth,torch.zeros_like(truth))
    return safe_p-safe_t

def body_dynamics(points,target,valid,scales):
    err=paired_error(points,target,valid)
    scales=torch.as_tensor(scales,device=points.device,dtype=points.dtype)
    if scales.shape!=(2,5) or not torch.isfinite(scales).all() or (scales<=0).any():raise ValueError("Positive D1/D2 per-part physical scales")
    losses=[];counts={}
    for k in (1,2):
        delta=torch.diff(err,n=k,dim=1)*30**k;mask=derivative_support(valid,k)
        groups=[]
        for i,g in enumerate(PARTS):
            m=mask[:,:,g];count=int(m.sum());counts[f"D{k}_part{i}"]=count
            if count:
                value=F.smooth_l1_loss(delta[:,:,g]/scales[k-1,i],torch.zeros_like(delta[:,:,g]),reduction="none",beta=1.)
                groups.append(value[m].mean())
        if not groups:raise ValueError("No observed continuous derivative templates")
        losses.append(torch.stack(groups).mean())
    return sum(losses),dict(D1=losses[0],D2=losses[1]),counts

def error_scales(points,target,valid):
    err=paired_error(points,target,valid);rows=[]
    for k in (1,2):
        delta=torch.diff(err,n=k,dim=1)*30**k;mask=derivative_support(valid,k)
        values=[]
        for group in PARTS:
            m=mask[:,:,group]
            if not m.any():raise ValueError("Calibration requires observed templates per body part")
            values.append(delta[:,:,group][m].square().mean().sqrt())
        rows.append(torch.stack(values))
    return torch.stack(rows)

def partial_forward(rt,xt,music,source_packet,t,seed,body,points,cfg,scales=None):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    ts=torch.full((len(xt),),int(t),device=xt.device,dtype=torch.long)
    rt.c.clear_all();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    with torch.no_grad():teacher=rt.d.model(xt,music,None,ts,cond_drop_prob=.2)
    torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);rt.c.set_source(source_packet)
    pred=rt.d.model(xt,music,None,ts,cond_drop_prob=.2);rt.c.clear_all()
    base,parts=partial_loss(pred,body,points,teacher,rt.n,rt.s,cfg["partial_weights"],cfg["partial_geometry_beta_m"])
    raw=rt.n.unnormalize(pred.clone());fk=fk22(raw,rt.s);core=fk[:,64:160]
    dyn=None;dynamic_parts={};counts={}
    if scales is not None:dyn,dynamic_parts,counts=body_dynamics(core,points,torch.ones_like(points[...,0],dtype=torch.bool),scales)
    return base,parts,dyn,dynamic_parts,pred,fk,counts

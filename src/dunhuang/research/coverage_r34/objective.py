"""Mask-consumed body targets: real source supervision replaces only the requested teacher region."""
import torch
import torch.nn.functional as F
from dunhuang.research.control_r33.contracts import TargetSupport
from dunhuang.research.control_r33.objective import body_dynamics,derivative_support
from dunhuang.research.reference_r26.geometry import fk22,relations

def region_loss(pred,body,points,teacher,normalizer,skeleton,weights,scales,valid=None,request=None):
    b,t=body.shape[:2]
    if pred.shape!=(b,256,139) or body.shape!=(b,256,132) or points.shape!=(b,256,22,3) or teacher.shape!=pred.shape:raise ValueError("Full state and actual observed body geometry required")
    valid=torch.ones(b,t,22,dtype=torch.bool,device=pred.device) if valid is None else valid
    request=torch.ones(b,t,dtype=torch.bool,device=pred.device) if request is None else request
    values=torch.full_like(pred,float("nan"));values[:,:,7:]=body
    observed=torch.zeros_like(pred,dtype=torch.bool);observed[:,:,7:]=valid.repeat_interleave(6,-1)
    editable=torch.ones_like(observed);support=TargetSupport(values,observed,editable).validate()
    requested=torch.zeros_like(observed);requested[:,:,7:]=request[:,:,None]
    mask=observed&requested&editable
    safe_target=torch.where(observed,values,torch.zeros_like(values))
    def masked_mean(x,m):
        m=m.expand_as(x);return x[m].mean() if m.any() else x.sum()*0
    error=pred-safe_target
    body_loss=masked_mean(error[:,:,7:].square(),mask[:,:,7:])
    vm=mask[:,1:,7:]&mask[:,:-1,7:]
    delta=(pred[:,1:,7:]-pred[:,:-1,7:])-(safe_target[:,1:,7:]-safe_target[:,:-1,7:])
    coordinate=masked_mean(delta.square(),vm)
    fk=fk22(normalizer.unnormalize(pred.clone()),skeleton)
    frame=valid.all(-1)&request
    target_points=torch.where(valid[:,:,:,None],points,torch.zeros_like(points))
    geometry=masked_mean(F.smooth_l1_loss(relations(fk),relations(target_points),beta=.05,reduction="none"),frame[:,:,None,None])
    # Keep the original first7 teacher coefficients, including its old outside contribution.
    teacher_error=(pred-teacher.detach()).square()
    outside=torch.ones_like(request);outside[:,64:160]=False
    keep=editable&~requested
    keep[:,:,:7]=outside[:,:,None]
    outside_loss=(teacher_error*keep).sum()/(b*160*139)
    terms=dict(body=body_loss,coordinate_difference=coordinate,local_geometry=geometry,prior_first7=teacher_error[:,:,:7].mean(),outside=outside_loss)
    base=sum(weights[k]*v for k,v in terms.items())
    dyn,dp,counts=body_dynamics(fk,target_points,valid&request[:,:,None],scales)
    return base,terms,dyn,dp,fk,counts

def forward(rt,xt,music,pkt,time,seed,body,points,scales):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    ts=torch.full((len(xt),),time,device=xt.device,dtype=torch.long)
    rt.c.clear_all();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    with torch.no_grad():teacher=rt.d.model(xt,music,None,ts,cond_drop_prob=.2)
    torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);rt.c.set_source(pkt)
    pred=rt.d.model(xt,music,None,ts,cond_drop_prob=.2);rt.c.clear_all()
    from .common import read
    weights=read("src/dunhuang/research/content_r32/config.json")["partial_weights"]
    base,parts,dyn,dp,fk,counts=region_loss(pred,body,points,teacher,rt.n,rt.s,weights,scales)
    return base,parts,dyn,dp,pred,fk,counts

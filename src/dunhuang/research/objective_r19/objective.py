"""Explicit masks/counts, SI derivatives and local seam/world supervision."""
import torch
from dunhuang.research.global_clock_r17.model import fk

PHYSICS=("seam_velocity","seam_acceleration","foot_position","foot_velocity")

def masked(error,mask):
    count=int(mask.sum())
    if count==0:return error.sum()*0,0
    return error.square()[mask].mean(),count

def reconstruction(pred,target,known):
    terms={};counts={}
    for prefix,mask in (("unknown",~known),("observed",known)):
        for kind,sl in (("root",slice(4,7)),("rotation",slice(7,151))):
            terms[prefix+"_"+kind],counts[prefix+"_"+kind]=masked(pred[:,:,sl]-target[:,:,sl],mask[:,:,sl])
        terms[prefix]=.5*(terms[prefix+"_root"]+terms[prefix+"_rotation"])
    return terms,counts

def scopes(known):
    # A Root derivative touching any unknown frame is eligible. Seam terms
    # are LOCAL to the fixed known/unknown edges8 and217, not the whole song.
    unknown=(~known[:,:,4:7]).any(-1)
    velocity=unknown[:,:-1]|unknown[:,1:]
    acceleration=unknown[:,:-2]|unknown[:,1:-1]|unknown[:,2:]
    device=known.device
    vtime=torch.arange(224,device=device)+.5
    atime=torch.arange(223,device=device)+1.
    seam_v=((abs(vtime-8)<=4)|(abs(vtime-217)<=4))[None]&velocity
    seam_a=((abs(atime-8)<=4)|(abs(atime-217)<=4))[None]&acceleration
    return unknown,velocity,seam_v,seam_a

def physical(pred,target,value,known,hard,scale,bias,origin,parents,offsets,scales):
    candidate=torch.where(hard,value,pred);raw=(candidate-bias)/scale;gt=(target-bias)/scale
    editable,velocity,seam_v,seam_a=scopes(known)
    terms={};counts={}
    for name,n,region in (("seam_velocity",1,seam_v),("seam_acceleration",2,seam_a)):
        residual=(torch.diff(raw[:,:,4:7],n=n,dim=1)-torch.diff(gt[:,:,4:7],n=n,dim=1))*30**n/scales[name]
        terms[name],counts[name]=masked(residual,region[:,:,None].expand_as(residual))
    points=fk(candidate,scale,bias,origin,parents,offsets)[:,:,[7,8,10,11]]
    truth=fk(target,scale,bias,origin,parents,offsets)[:,:,[7,8,10,11]]
    residual=(points-truth)/scales["foot_position"]
    terms["foot_position"],counts["foot_position"]=masked(residual,editable[:,:,None,None].expand_as(residual))
    residual=(torch.diff(points,dim=1)-torch.diff(truth,dim=1))*30/scales["foot_velocity"]
    terms["foot_velocity"],counts["foot_velocity"]=masked(residual,velocity[:,:,None,None].expand_as(residual))
    return terms,counts

def geometry(pred,scale,bias):
    with torch.no_grad():
        r=((pred-bias)/scale)[:,:,7:].reshape(-1,24,6)
        a,b=r[...,:3],r[...,3:];norm=a.norm(dim=-1)
        u=torch.nn.functional.normalize(a,dim=-1)
        perpendicular=(b-(u*b).sum(-1,keepdim=True)*u).norm(dim=-1)
        def stats(x):return dict(min=float(x.min()),p01=float(torch.quantile(x.flatten(),.01)),
            median=float(x.median()),below_1e_3=int((x<1e-3).sum()),elements=x.numel())
        return dict(first_vector=stats(norm),perpendicular_vector=stats(perpendicular))

def static_weights(norms,reference,budget=.1,floor=1e-12):
    """No reciprocal inflation: each physical weight is bounded above budget/4."""
    result={}
    for name in PHYSICS:
        ratios=[norms[name][g]/max(reference[g],floor) for g in ("motion","head")]
        result[name]=budget/(len(PHYSICS)*max(1.,*ratios))
    return result

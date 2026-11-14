"""Paired whole-body velocity/acceleration error, not motion-to-zero damping."""
import torch
from dunhuang.research.global_clock_r17.model import fk
from dunhuang.research.motion_bridge.model import rotations
from dunhuang.research.objective_r19.objective import scopes
GROUPS=dict(root=[0],lower=[1,2,4,5,7,8,10,11],trunk=[3,6,9,12,15],upper=[13,14,16,17,18,19,20,21,22,23])

def temporal(pred,y,v,k,h,cache,origin,scales):
    a=fk(torch.where(h,v,pred),cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])
    b=fk(y,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])
    editable,velocity,seam_v,seam_a=scopes(k)
    acceleration=editable[:,:-2]|editable[:,1:-1]|editable[:,2:]
    terms={};counts={}
    for name,n,base_mask,seam_mask in (("body_velocity",1,velocity,seam_v),("body_acceleration",2,acceleration,seam_a)):
        pieces=[]
        for group,joints in GROUPS.items():
            ids=[j for j in joints if not(n==1 and j in (7,8,10,11))]
            mask=base_mask&~seam_mask if group=="root" else base_mask
            error=(torch.diff(a[:,:,ids],n=n,dim=1)-torch.diff(b[:,:,ids],n=n,dim=1))*30**n/scales[name]
            use=mask[:,:,None,None].expand_as(error);count=int(use.sum())
            if count:pieces.append(torch.nn.functional.huber_loss(error[use],torch.zeros_like(error[use]),delta=1.))
            counts[name+"_"+group]=count
        terms[name]=torch.stack(pieces).mean()
    return terms,counts

def stage_metrics(x,cache,origin,start=0):
    x=x.detach();raw=(x-cache["bias"])/cache["scale"];r=raw[:,:,7:].reshape(-1,24,6)
    a,b=r[...,:3],r[...,3:];u=torch.nn.functional.normalize(a,dim=-1)
    na=a.norm(dim=-1);nb=(b-(u*b).sum(-1,keepdim=True)*u).norm(dim=-1)
    good=torch.isfinite(x).all() and bool((na>=1e-6).all()) and bool((nb>=1e-6).all())
    base=dict(valid_rotation=bool(good),first_vector_min=float(na.min()) if torch.isfinite(na).all() else None,perpendicular_min=float(nb.min()) if torch.isfinite(nb).all() else None)
    if not good:return dict(base,fk_jerk=None)
    p=fk(x,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])
    jerk=(torch.diff(p,n=3,dim=1)*30**3).norm(dim=-1)[0]
    times=torch.arange(len(jerk),device=x.device)+start
    masks={}
    masks["fixed"]=sum(((times<c)&(times+3>=c)).int() for c in (8,217))>0
    masks["fusion"]=(sum(((times<c)&(times+3>=c)).int() for c in (75,150))>0)&~masks["fixed"]
    masks["phase"]=((times<112)&(times+3>=112))&~masks["fixed"]&~masks["fusion"]
    masks["interior"]=~(masks["fixed"]|masks["fusion"]|masks["phase"])
    buckets={}
    for name,mask in masks.items():
        q=jerk[mask];buckets[name]=dict(templates=int(mask.sum()),max=float(q.max()),p95=float(torch.quantile(q.flatten(),.95))) if q.numel() else None
    i=int(jerk.argmax());frame=i//24;tip=i%24;chain=[];joint=tip
    while joint>=0:
        local=raw[0,:,7+joint*6:13+joint*6]
        index=min(frame+1,len(local)-2)
        R=rotations(local[None]);rel=R[:,1:].transpose(-1,-2)@R[:,:-1]
        angle=torch.acos(((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))
        chain.append(dict(joint=joint,local_angular_step_max_rad=float(angle[0,frame:frame+3].max()),local6d_step=float((local[index+1]-local[index]).norm())))
        joint=cache["parents"][joint]
    return dict(base,fk_jerk=dict(max=float(jerk.max()),p95=float(torch.quantile(jerk.flatten(),.95)),
        peak_joint=tip,peak_frame=float(start+frame+1.5),buckets=buckets,ancestor_local_proxy=chain),
        activity=float((torch.diff(p,dim=1)*30).norm(dim=-1).mean()))

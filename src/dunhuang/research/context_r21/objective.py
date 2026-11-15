"""Same A2 losses, or equal-mass observation-crossing/editable template means."""
import torch
from dunhuang.research.global_clock_r17.model import fk
from dunhuang.research.temporal_r20.objective import GROUPS
from dunhuang.research.objective_r19.objective import reconstruction
from .contract import regions

def scopes(known):
    observed=known[:,:,4:7].all(-1);editable=~observed
    velocity=editable[:,:-1]|editable[:,1:];acceleration=editable[:,:-2]|editable[:,1:-1]|editable[:,2:]
    n=known.shape[1];vt=torch.arange(n-1,device=known.device)+.5;at=torch.arange(n-2,device=known.device)+1.
    sv=torch.zeros_like(velocity);sa=torch.zeros_like(acceleration)
    for batch in range(len(known)):
        cuts=(observed[batch,1:]!=observed[batch,:-1]).nonzero().flatten()+1
        for cut in cuts:sv[batch]|=abs(vt-cut)<=4;sa[batch]|=abs(at-cut)<=4
    return editable,velocity,acceleration,sv&velocity,sa&acceleration

def masked_mean(error,mask,huber=False):
    use=mask
    while use.ndim<error.ndim:use=use.unsqueeze(-1)
    use=use.expand_as(error);count=int(use.sum())
    loss=torch.nn.functional.huber_loss(error[use],torch.zeros_like(error[use]),delta=1.) if huber and count else error[use].square().mean() if count else error.sum()*0
    return loss,count

def organized(error,eligible,observed,order,balanced,huber=False):
    masks=regions(observed,order);details={};pieces=[]
    for label in ("crossing","editable"):
        loss,count=masked_mean(error,eligible&masks[label],huber);details[label]=count
        if count:pieces.append(loss)
    if balanced:return torch.stack(pieces).mean() if pieces else error.sum()*0,details
    loss,count=masked_mean(error,eligible,huber);return loss,details

def losses(pred,y,value,known,hard,cache,origin,balanced,proxy=None):
    basic,counts=reconstruction(pred,y,known);terms=dict(basic)
    candidate=torch.where(hard,value,pred);raw=(candidate-cache["bias"])/cache["scale"];gt=(y-cache["bias"])/cache["scale"]
    edit,velocity,acceleration,sv,sa=scopes(known);observed=known[:,:,4:7].all(-1)
    for name,n,mask in (("seam_velocity",1,sv),("seam_acceleration",2,sa)):
        error=(torch.diff(raw[:,:,4:7],n=n,dim=1)-torch.diff(gt[:,:,4:7],n=n,dim=1))*30**n/(3. if n==1 else 30.)
        terms[name],counts[name]=masked_mean(error,mask)
    points=fk(candidate,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])
    truth=fk(y,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])
    foot=points[:,:,[7,8,10,11]];target=truth[:,:,[7,8,10,11]]
    terms["foot_position"],counts["foot_position"]=masked_mean((foot-target)/.1,edit)
    error=(torch.diff(foot,dim=1)-torch.diff(target,dim=1))*30
    terms["foot_velocity"],counts["foot_velocity"]=organized(error,velocity,observed,1,balanced)
    for name,n,mask,seam,scale in (("body_velocity",1,velocity,sv,3.),("body_acceleration",2,acceleration,sa,30.)):
        pieces=[]
        for group,joints in GROUPS.items():
            ids=[j for j in joints if not(n==1 and j in (7,8,10,11))]
            eligible=mask&~seam if group=="root" else mask
            residual=(torch.diff(points[:,:,ids],n=n,dim=1)-torch.diff(truth[:,:,ids],n=n,dim=1))*30**n/scale
            loss,c=organized(residual,eligible,observed,n,balanced,huber=True);pieces.append(loss);counts[name+"_"+group]=c
        terms[name]=torch.stack(pieces).mean()
    if proxy is not None:
        scales=cache["support_scales"];feet=points[:,:,cache["proxy_foot_ids"]];fixed=hard[:,:,4:].all(-1);editable=~fixed
        static=proxy["proxy_static"];segment=proxy["proxy_segment"]
        use=static[:,1:]&static[:,:-1]&(segment[:,1:]==segment[:,:-1])&(editable[:,1:]|editable[:,:-1])[:,:,None]
        terms["support_velocity"],counts["support_velocity"]=masked_mean(torch.diff(feet,dim=1)*30/scales["support_velocity"],use,huber=True)
        terms["support_anchor"],counts["support_anchor"]=masked_mean((feet-proxy["proxy_anchor"])/scales["support_anchor"],static&editable[:,:,None],huber=True)
        error=(proxy["proxy_floor"][:,None,None]+scales["swing_clearance"]-feet[:,:,:,1]).clamp_min(0)/scales["swing_clearance"]
        terms["swing_clearance"],counts["swing_clearance"]=masked_mean(error,proxy["proxy_swing"]&editable[:,:,None],huber=True)
    return terms,counts

def build(cfg,predict,cache,s,old_weights,temporal_weights,support_weights=None,balanced=False,proxy=None):
    y,m,o,v,k,h,meta,t,xt=s;pred,_=predict(xt,m,v,k,meta,t)
    terms,counts=losses(pred,y,v,k,h,cache,o,balanced,proxy)
    active=dict(unknown=terms["unknown"],observed=.1*terms["observed"])
    active.update({n:w*terms[n] for n,w in dict(old_weights,**temporal_weights,**(support_weights or {})).items()})
    return pred,terms,counts,active,sum(active.values())

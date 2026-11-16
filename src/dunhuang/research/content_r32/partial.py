"""Partial BODY observation loss on complete, model-visited states; never forms x0."""
import torch
import torch.nn.functional as F
from dunhuang.research.reference_r26.geometry import fk22,relations

def partial_loss(pred,body,local_points,no_ref,normalizer,skeleton,weights,beta=.05):
    if pred.ndim!=3 or pred.shape[1:]!=(256,139) or no_ref.shape!=pred.shape:
        raise ValueError("Complete model state and same-platform NoRef prediction required")
    if body.shape!=(len(pred),96,132) or local_points.shape!=(len(pred),96,22,3):
        raise ValueError("Only observed96x132 body and local22 geometry accepted")
    if not torch.isfinite(body).all() or not torch.isfinite(local_points).all():
        raise ValueError("Observed body/geometry must be finite; UNKNOWN is not zero-filled")
    if (local_points[:,:,:1].abs()>1e-5).any():
        raise ValueError("Target geometry must be Root-relative, not an unknown world trajectory")
    target=body
    p=pred[:,64:160,7:]
    points=fk22(normalizer.unnormalize(pred.clone()),skeleton)[:,64:160]
    terms=dict(body=(p-target).square().mean(),
        coordinate_difference=(torch.diff(p,dim=1)-torch.diff(target,dim=1)).square().mean(),
        local_geometry=F.smooth_l1_loss(relations(points),relations(local_points),beta=beta),
        prior_first7=(pred[:,:,:7]-no_ref[:,:,:7].detach()).square().mean(),
        outside=torch.cat((pred[:,:64]-no_ref[:,:64].detach(),pred[:,160:]-no_ref[:,160:].detach()),1).square().mean())
    return sum(weights[k]*v for k,v in terms.items()),terms

def partial_objective(d,b,xt,music,ref,t,seed,body,points,n,s,cfg):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    ts=torch.full((len(xt),),int(t),device=xt.device,dtype=torch.long)
    b.clear();rng=torch.cuda.get_rng_state();cpu=torch.get_rng_state()
    with torch.no_grad():base=d.model(xt,music,None,ts,cond_drop_prob=.2)
    torch.cuda.set_rng_state(rng);torch.set_rng_state(cpu);b.set_reference(**ref)
    pred=d.model(xt,music,None,ts,cond_drop_prob=.2);b.clear()
    value,parts=partial_loss(pred,body,points,base,n,s,cfg["partial_weights"],cfg["partial_geometry_beta_m"])
    return value,{k:float(v.detach()) for k,v in parts.items()}

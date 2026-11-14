"""Loss scopes, branch cancellation and actual-update directional readouts."""
import torch
from dunhuang.research.global_clock_r17.objective import losses as legacy_losses
from dunhuang.research.task_denoising_r18.engine import amplitude,unknown_mse
from .objective import reconstruction,physical,geometry
from .common import gradient,group_norm,parameters

def build(cfg,predict,cache,s,arm,weights,ranking):
    y,m,origin,v,k,h,meta,t,xt=s
    pred,_=predict(xt,m,v,k,meta,t)
    simple,counts=reconstruction(pred,y,k)
    phys,pcounts=physical(pred,y,v,k,h,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"],cfg["physics_scales"])
    terms=dict(simple,**phys)
    if arm=="N0":
        old=legacy_losses(pred,y,v,k,h,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])
        old["amplitude"]=amplitude(pred,y,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])
        old["context"]=pred.sum()*0
        if ranking:
            wrong,_=predict(xt,m,v,k,meta,t,"content")
            old["context"]=torch.relu(.001+unknown_mse(pred,y,k)-unknown_mse(wrong,y,k))
        active={name:factor*old[name] for name,factor in dict(reconstruction=1,boundary=.1,world_fk=.1,amplitude=.1,context=.05).items()}
    else:
        active=dict(unknown=simple["unknown"],observed=cfg["observed_weight"]*simple["observed"])
        if arm in ("N2","U"):active.update({name:weights[name]*phys[name] for name in weights})
    return pred,terms,dict(counts,**pcounts),active,sum(active.values())

def analyze(cfg,w,head,predict,cache,s,pred,terms,counts,active):
    y,m,o,v,k,h,meta,t,xt=s;params=parameters(w,head);split=sum(p.numel() for p in w.selected())
    gu=gradient(terms["unknown"],params)
    gradients={name:gradient(value,params) for name,value in active.items()}
    norms={name:group_norm(g,split) for name,g in gradients.items()}
    cancellation={}
    ec=unknown_mse(pred,y,k);gc=gradient(ec,params)
    for control in ("content","typed_content"):
        wrong,_=predict(xt,m,v,k,meta,t,control)
        ew=unknown_mse(wrong,y,k);gw=gradient(ew,params)
        nc=group_norm(gc,split);nw=group_norm(gw,split);nd=group_norm(gc-gw,split)
        cancellation[control]=dict(correct_error=float(ec.detach()),wrong_error=float(ew.detach()),
            correct_gradient=nc,wrong_gradient=nw,difference_gradient=nd,
            difference_over_sum={g:nd[g]/max(nc[g]+nw[g],1e-30) for g in nc})
    record=dict(unweighted={n:float(v.detach()) for n,v in terms.items()},effective_counts=counts,
        actual_weighted_gradient_norms=norms,unknown_gradient=group_norm(gu,split),
        branch_cancellation=cancellation,geometry=geometry(pred,cache["scale"],cache["bias"]))
    return record,gu

def backward(w,head,objective):
    for p in parameters(w,head):p.grad=None
    objective.backward()
    a=float(torch.nn.utils.clip_grad_norm_(w.selected(),1.,error_if_nonfinite=True))
    b=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True))
    assert all(p.grad is None for p in w.parameters() if not p.requires_grad)
    return dict(motion_norm=a,head_norm=b,motion_clip_factor=min(1.,1/(a+1e-6)),head_clip_factor=min(1.,1/(b+1e-6)))

def after_update(record,gu,delta,predict,s):
    y,m,o,v,k,h,meta,t,xt=s
    with torch.no_grad():
        pred,_=predict(xt,m,v,k,meta,t);r,_=reconstruction(pred,y,k)
    before=record["unweighted"]["unknown"];after=float(r["unknown"])
    return dict(record,actual_update_norm=float(delta.norm()),unknown_gradient_dot_update=float(gu@delta),
        unknown_before=before,unknown_after=after,unknown_change=after-before,
        first_order_predicts_improvement=bool(gu@delta<0),actual_improvement=after<before)

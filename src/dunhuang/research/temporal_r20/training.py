"""Shared losses/actual-update observer, no changes to inherited CFG/clamp."""
import torch
from dunhuang.research.objective_r19.objective import reconstruction,physical,geometry
from dunhuang.research.objective_r19.common import parameters,gradient,group_norm
from .objective import temporal

def build(cfg,predict,cache,s,old_weights,new_weights=None):
    y,m,o,v,k,h,meta,t,xt=s;pred,_=predict(xt,m,v,k,meta,t)
    basic,counts=reconstruction(pred,y,k)
    ph,pc=physical(pred,y,v,k,h,cache["scale"],cache["bias"],o,cache["parents"],cache["offsets"],
        dict(seam_velocity=3.,seam_acceleration=30.,foot_position=.1,foot_velocity=1.))
    terms=dict(basic,**ph);counts.update(pc)
    active=dict(unknown=basic["unknown"],observed=.1*basic["observed"],**{n:old_weights[n]*ph[n] for n in old_weights})
    if new_weights:
        extra,ec=temporal(pred,y,v,k,h,cache,o,cfg["temporal_scales"])
        terms.update(extra);counts.update(ec);active.update({n:new_weights[n]*extra[n] for n in new_weights})
    return pred,terms,counts,active,sum(active.values())

def backward(w,head,objective):
    params=parameters(w,head)
    for p in params:p.grad=None
    objective.backward()
    raw=parameter_stats(w,head)
    a=float(torch.nn.utils.clip_grad_norm_(w.selected(),1.,error_if_nonfinite=True))
    b=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True))
    assert all(p.grad is None for p in w.parameters() if not p.requires_grad)
    return dict(motion_norm=a,head_norm=b,parameter_unclipped=raw)

def parameter_stats(w,head,delta=None):
    named=[(n,p) for n,p in w.net.named_parameters() if p.requires_grad]+[("head."+n,p) for n,p in head.named_parameters()]
    prefixes=dict(input="input_projection.",decoder_first="seqTransDecoder.stack.0.",decoder_last="seqTransDecoder.stack.7.",output="final_layer.",state="head.")
    result={};ranges=[];index=0
    for n,p in named:ranges.append((n,p,index,index+p.numel()));index+=p.numel()
    for label,prefix in prefixes.items():
        ps=[(p,a,b) for n,p,a,b in ranges if n.startswith(prefix)]
        gs=[p.grad.detach().square().sum() for p,a,b in ps if p.grad is not None]
        result[label]=dict(trainable_elements=sum(p.numel() for p,a,b in ps),
            gradient_norm=float(torch.stack(gs).sum().sqrt()) if gs else 0.,
            update_norm=float(torch.stack([delta[a:b].square().sum() for p,a,b in ps]).sum().sqrt()) if ps and delta is not None else None)
    return result

def observe_before(w,head,pred,terms,counts,active,cache):
    ps=parameters(w,head);split=sum(p.numel() for p in w.selected())
    gu=gradient(terms["unknown"],ps)
    record=dict(unknown_before=float(terms["unknown"].detach()),parts={n:float(v.detach()) for n,v in terms.items()},
        counts=counts,weighted_gradients={n:group_norm(gradient(v,ps),split) for n,v in active.items()},
        geometry=geometry(pred,cache["scale"],cache["bias"]))
    return record,gu

def observe_after(record,gu,delta,predict,s,w,head,layer_forward,layer_backward,clipping):
    y,m,o,v,k,h,meta,t,xt=s
    with torch.no_grad():
        p,_=predict(xt,m,v,k,meta,t);terms,_=reconstruction(p,y,k)
    return dict(record,unknown_after=float(terms["unknown"]),unknown_gradient_dot_update=float(gu@delta),
        actual_update_norm=float(delta.norm()),parameter_stats=parameter_stats(w,head,delta),
        layers_forward=layer_forward,layers_backward=layer_backward,clipping=clipping)

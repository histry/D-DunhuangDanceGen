"""Independent branch reconstruction; physical terms keep guided/clip/hard input."""
import torch
from dunhuang.research.context_r21.objective import losses
from dunhuang.research.objective_r19.objective import reconstruction
from dunhuang.research.temporal_r20.trace import blend

def branch_terms(raw,target,known,clipped):
    assert raw.requires_grad or not torch.is_grad_enabled(),"Detached branch cannot train"
    return reconstruction(blend(raw.clamp(-1,1) if clipped else raw),target,known)[0]

def supervise(arm,unc,cond,guided,target,known):
    if arm=="G0":return reconstruction(guided,target,known)[0]
    if arm not in ("G1","G2"):raise ValueError("Unknown registered arm")
    u=branch_terms(unc,target,known,arm=="G1");c=branch_terms(cond,target,known,arm=="G1")
    return {n:.25*u[n]+.75*c[n] for n in u}

def build(cfg,predict,cache,s,old_weights,temp_weights,arm):
    y,m,o,v,k,h,meta,t,xt=s
    predict.capture_branches=True;pred,_=predict(xt,m,v,k,meta,t)
    assert len(predict.branches)==2
    basic,counts=losses(pred,y,v,k,h,cache,o,True)
    supervised=supervise(arm,*predict.branches,pred,y,k)
    terms=dict(basic);terms.update(supervised)
    terms["guided_unknown"]=basic["unknown"];terms["guided_observed"]=basic["observed"]
    active=dict(unknown=terms["unknown"],observed=.1*terms["observed"])
    active.update({n:w*basic[n] for n,w in dict(old_weights,**temp_weights).items()})
    return pred,terms,counts,active,sum(active.values())

def branch_scores(predict,target,known):
    result={}
    assert len(predict.branches)==2
    for name,raw in zip(("unconditional_music","conditional_music"),predict.branches):
        result[name]={}
        for label,clipped in (("raw",False),("clipped",True)):
            terms=reconstruction(blend(raw.clamp(-1,1) if clipped else raw),target,known)[0]
            result[name][label]={n:float(terms[n].detach()) for n in ("unknown","observed")}
    return result

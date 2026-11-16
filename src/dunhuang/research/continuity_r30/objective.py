"""Physical-unit paired dynamics through the actual clipped decoder."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from dunhuang.research.contract_r28.train import objective as original_objective
from dunhuang.research.reference_r26.geometry import fk22
from dunhuang.research.lodge_adapter_r25.model import PARTS

def physical(points,target):
    root=points[:,64:160,:1];tr=target[:,64:160,:1]
    rel=points[:,64:160]-root;truth=target[:,64:160]-tr
    r=[];b=[]
    for k,beta in ((1,.5),(2,5.)):
        diff=lambda x:torch.diff(x,n=k,dim=1)*30**k
        r.append(F.smooth_l1_loss(diff(root),diff(tr),beta=beta))
        p,q=diff(rel),diff(truth)
        b.append(torch.stack([F.smooth_l1_loss(p[:,:,g],q[:,:,g],beta=beta) for g in PARTS]).mean())
    return sum(r),sum(b)

@contextmanager
def conditional_prediction(d,a):
    old=d.model.forward;found=[]
    def capture(*args,**kwargs):
        value=old(*args,**kwargs)
        if a.context is not None and torch.is_grad_enabled():found.append(value)
        return value
    d.model.forward=capture
    try:yield found
    finally:d.model.forward=old

def objective(d,a,y,music,ref,t,seed,n=None,skeleton=None,target=None,weights=None,inspect=False):
    if weights is None and not inspect:return original_objective(d,a,y,music,ref,t,seed)
    with conditional_prediction(d,a) as captures:
        base,parts=original_objective(d,a,y,music,ref,t,seed)
    assert len(captures)==1
    pred=captures[0];raw=n.unnormalize(pred.clone());points=fk22(raw,skeleton)
    root,body=physical(points,target)
    if inspect:return base,parts,pred,raw,points,root,body
    loss=base+weights["root"]*root+weights["body"]*body
    return loss,dict(parts,physical_Root=float(root.detach()),physical_body=float(body.detach()))

@contextmanager
def suppress_direct_root_delta(d,a):
    """Same-state, same-CFG-branch counterfactual; C4/body retained, no fixed Root trajectory."""
    old=d.model.forward;counts={"branch_calls":0,"NoRef_counterfactual_calls":0}
    def forward(*args,**kwargs):
        counts["branch_calls"]+=1
        context=a.context
        if context is None or not context[2].any():return old(*args,**kwargs)
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        ref=old(*args,**kwargs);after_cpu=torch.get_rng_state();after_gpu=torch.cuda.get_rng_state()
        try:
            a.context=None;torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
            with torch.no_grad():base=old(*args,**kwargs)
        finally:
            a.context=context;torch.set_rng_state(after_cpu);torch.cuda.set_rng_state(after_gpu)
        counts["NoRef_counterfactual_calls"]+=1
        out=ref.clone();out[:,:,4:7]=base[:,:,4:7]
        assert torch.equal(out[:,:,:4],ref[:,:,:4]) and torch.equal(out[:,:,7:],ref[:,:,7:])
        return out
    d.model.forward=forward
    try:yield counts
    finally:d.model.forward=old

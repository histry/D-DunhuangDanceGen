"""Observe the original sampler without changing its update, CFG, or clamping."""
import hashlib
from contextlib import contextmanager
import numpy as np
import torch

@contextmanager
def row_noise(seeds,frames=256,dim=139,device="cuda"):
    original=torch.randn;like=torch.randn_like;shape=(len(seeds),frames,dim)
    generators=[torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    digests=[hashlib.sha256() for s in seeds];receipt=dict(seeds=list(seeds),draws=0)
    def draw(dtype=torch.float32):
        rows=[original((frames,dim),generator=g,device=device,dtype=dtype) for g in generators]
        for h,x in zip(digests,rows):h.update(x.detach().cpu().numpy().tobytes())
        receipt["draws"]+=1
        return torch.stack(rows)
    def randn(*args,**kwargs):
        requested=tuple(args[0]) if len(args)==1 and isinstance(args[0],(tuple,list,torch.Size)) else tuple(args)
        return draw(kwargs.get("dtype",torch.float32)) if requested==shape else original(*args,**kwargs)
    def randn_like(x,*args,**kwargs):
        return draw(kwargs.get("dtype",x.dtype)) if tuple(x.shape)==shape else like(x,*args,**kwargs)
    torch.randn=randn;torch.randn_like=randn_like
    try:yield receipt
    finally:
        torch.randn=original;torch.randn_like=like
        receipt["row_sha256"]=[h.hexdigest() for h in digests]

@contextmanager
def record_sampler(diffusion,selected=(999,519,119,19),target=None):
    old_predictions=diffusion.model_predictions;old_forward=diffusion.model.forward
    capture=dict(states=[],times=[],selected={},branches={},errors=[],saturation=[])
    def forward(*args,**kwargs):
        value=old_forward(*args,**kwargs)
        probability=kwargs.get("cond_drop_prob",0)
        if probability in (0,1):capture["branches"]["unc" if probability==1 else "cond"]=value.detach().clone()
        return value
    def predictions(x,cond,genre,t,*args,**kwargs):
        time=int(t[0]);assert (t==time).all()
        capture["states"].append(x.detach().cpu().numpy().copy());capture["times"].append(time);capture["branches"]={}
        value=old_predictions(x,cond,genre,t,*args,**kwargs)
        weight=kwargs.get("weight");weight=diffusion.guidance_weight if weight is None else weight
        unc=capture["branches"]["unc"];conditional=capture["branches"]["cond"]
        raw=unc+(conditional-unc)*weight
        expected=raw.clamp(-1,1) if kwargs.get("clip_x_start",False) else raw
        assert torch.equal(expected,value[1]),"Observed CFG/clamp must exactly match official output"
        if target is not None:
            groups=[slice(0,4),slice(4,7),slice(7,139)]
            indices=[torch.arange(64,160,device=x.device),torch.cat((torch.arange(64,device=x.device),torch.arange(160,256,device=x.device)))]
            errors=torch.stack([torch.stack([torch.stack([(p[:,region,g]-target[:,region,g]).square().mean((1,2)) for region in indices],-1) for g in groups],-2) for p in [unc,conditional,raw,value[1]]],-3)
            saturated=torch.stack([torch.stack([(raw[:,region,g].abs()>1).float().mean((1,2)) for region in indices],-1) for g in groups],-2)
            capture["errors"].append(errors.detach().cpu().numpy());capture["saturation"].append(saturated.detach().cpu().numpy())
        if time in selected:
            capture["selected"][time]={k:v.detach().cpu().numpy() for k,v in dict(unc=unc,cond=conditional,cfg=raw,clipped=value[1],pred_noise=value[0]).items()}
            capture["selected"][time]["weight"]=float(weight)
        return value
    diffusion.model.forward=forward;diffusion.model_predictions=predictions
    try:yield capture
    finally:
        diffusion.model.forward=old_forward;diffusion.model_predictions=old_predictions
        capture.pop("branches",None)

def sample(d,adapter,reference,music,seeds,*,trace=False,selected=(999,519,119,19),target=None):
    if len(seeds)<2 or len(seeds)!=len(music):raise ValueError("Native soft sampler requires batch>=2")
    adapter.clear()
    if reference is not None:
        with torch.no_grad():adapter.set_reference(**reference)
    torch.manual_seed(27041);torch.cuda.manual_seed_all(27041)
    zeros=torch.zeros(len(seeds),256,139,device=music.device)
    if trace:
        with torch.no_grad(),row_noise(seeds,device=str(music.device)) as noise,record_sampler(d,selected,target) as record:
            value=d.inpaint_soft_ddim(tuple(zeros.shape),music,None,constraint=dict(value=zeros,mask=zeros))
    else:
        with torch.no_grad(),row_noise(seeds,device=str(music.device)) as noise:
            value=d.inpaint_soft_ddim(tuple(zeros.shape),music,None,constraint=dict(value=zeros,mask=zeros))
        record=None
    adapter.clear();assert noise["draws"]==50
    return value,noise,record

def trace_arrays(record):
    arrays=dict(xt=np.stack(record["states"]),times=np.array(record["times"],np.int64))
    if record["errors"]:arrays["errors"]=np.stack(record["errors"]);arrays["saturation"]=np.stack(record["saturation"])
    for t,row in record["selected"].items():
        for k,v in row.items():arrays[f"p_{t}_{k}"]=np.asarray(v)
    return arrays

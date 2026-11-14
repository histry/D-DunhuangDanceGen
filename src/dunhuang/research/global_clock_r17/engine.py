"""Frozen R13 local views and global225 reverse diffusion; no 225-token EDGE call."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked
from dunhuang.research.source_capacity.common import read
from dunhuang.research.masked_denoiser.common import setup,make,checkpoint_load,tensor_hash
from .model import Predictor,SharedState
from .contract import conditions,metadata

def load(cfg,mode=None,checkpoint=None):
    _,_,_,_,_,data,base,diffusion=setup(read(cfg["r13_config"]))
    wrapper=make(read(cfg["r13_config"]),base)
    ref=read(Path(cfg["r13_root"])/"main/result.json")["checkpoint"]
    assets=read(Path(cfg["r16_root"])/"preflight/protocol.json")["assets"]
    checkpoint_load(read(cfg["r13_config"]),wrapper,ref,assets)
    for p in wrapper.parameters():p.requires_grad_(False)
    adapter=None
    if mode:
        torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
        adapter=SharedState(mode,cfg["state_width"]).cuda()
        if checkpoint:
            state=torch.load(checked(checkpoint),map_location="cpu",weights_only=True)
            assert state["config"]==cfg and state["mode"]==mode
            adapter.load_state_dict(state["state"])
    return wrapper,diffusion,data,Predictor(wrapper,adapter),adapter

def tensors(record,device="cuda"):
    z=np.load(checked(record["arrays"]))
    y=torch.tensor(z["encoded"][None],device=device);m=torch.tensor(np.stack((z["music"][:150],z["music"][75:])),device=device)
    return z,y,m

def observed(y,task,*,hard=True,prefix=None,boundary=112,extra=None):
    v,k,h=conditions(y[0].detach().cpu().numpy(),task,hard=hard,prefix=prefix,boundary=boundary,extra=extra)
    return (torch.tensor(v[None],device=y.device),torch.tensor(k[None],device=y.device),torch.tensor(h[None],device=y.device))

def global_sample(diffusion,predict,music,value,known,hard,meta,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);noise_hash=hashlib.sha256()
    def draw(x):
        noise_hash.update(x.detach().cpu().numpy().tobytes());return x
    x=draw(torch.randn_like(value));steps=[];controls={};last=None
    times=list(reversed(torch.linspace(-1,999,51).int().tolist()))
    with torch.no_grad():
        for step,(t,nxt) in enumerate(zip(times[:-1],times[1:])):
            ts=torch.tensor([t],device=x.device,dtype=torch.long)
            p,windows=predict(x,music,value,known,meta,ts)
            if step in (24,49):
                for control in ("correct","reversed","removed"):
                    q=p if control=="correct" or predict.adapter is None else predict(x,music,value,known,meta,ts,control)[0]
                    controls[str(t)+"_"+control]=q[0].cpu().numpy()
            clean=torch.where(hard,value,p)
            steps.append(dict(step=step,t=t,x_t_sha256=tensor_hash(x),raw_sha256=tensor_hash(p),
                raw_overlap_max=float(abs(windows[0,75:]-windows[1,:75]).max()),
                raw_known_rmse=float((p[known]-value[known]).square().mean().sqrt()) if known.any() else None))
            last=p
            if nxt<0:x=clean;continue
            alpha=diffusion.alphas_cumprod[t];na=diffusion.alphas_cumprod[nxt]
            eps=(x-alpha.sqrt()*clean)/(1-alpha).sqrt()
            sigma=((1-alpha/na)*(1-na)/(1-alpha)).sqrt()
            noise=draw(torch.randn_like(x))
            x=na.sqrt()*clean+(1-na-sigma**2).clamp(min=0).sqrt()*eps+sigma*noise
            x=torch.where(hard,na.sqrt()*value+(1-na).sqrt()*noise,x)
    return dict(sample=x[0].cpu().numpy(),raw_final=last[0].cpu().numpy(),
        condition=value[0].cpu().numpy(),known=known[0].cpu().numpy(),hard=hard[0].cpu().numpy(),**controls),dict(
        steps=steps,noise_sha256=noise_hash.hexdigest(),all_edge_calls_local150=True,global_noise_frames=225,
        final_hard_error=float(abs(x[hard]-value[hard]).max()) if hard.any() else None,output_pose_paste=False)

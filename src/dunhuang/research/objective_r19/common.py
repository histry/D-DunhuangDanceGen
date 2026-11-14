"""Sealed data, fixed schedule and unchanged R18 architecture initialization."""
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha
from dunhuang.research.source_capacity.common import read
from dunhuang.research.task_denoising_r18.engine import population as parent_population,load as parent_load
from dunhuang.research.task_denoising_r18.model import Predictor,memory_support
from dunhuang.research.global_clock_r17.engine import tensors,observed
from dunhuang.research.global_clock_r17.contract import metadata
from dunhuang.research.global_clock_r17.model import fk
from .model import State

def source_config(cfg):
    p=Path(cfg["parent"]);seal=read("reports/r18_receipt.json")
    assert sha(p/"p0/protocol.json")==seal["evidence"]["p0/protocol.json"]
    return read(p/"p0/protocol.json")["config"]

def population(cfg):
    pre,fit,ev=parent_population(source_config(cfg))
    return pre,[r for r in fit if r["case"] in cfg["cases"]]

def load(cfg,mode="self",reference=None,adapt=True):
    oldcfg=source_config(cfg)
    w,d,data,p,oldhead=parent_load(oldcfg,mode,adapt=adapt);p.close()
    head=State(mode,oldcfg["state_width"]).cuda();head.load_state_dict(oldhead.state_dict())
    if reference:restore(cfg,w,head,reference)
    return w,d,data,Predictor(w,head),head

def restore(cfg,w,head,reference):
    state=torch.load(checked(reference),map_location="cpu",weights_only=True)
    assert state["config"]==cfg
    assert state["mode"]==head.memory_mode
    w.load_selected(state["motion"]);head.load_state_dict(state["head"])

def capture(cfg,w,head,path,arm,step):
    torch.save(dict(schema=cfg["schema"],config=cfg,mode=head.memory_mode,arm=arm,step=step,
        motion=w.state(),head={n:p.detach().cpu().clone() for n,p in head.state_dict().items()},
        operational_qualified=False),path)
    from dunhuang.data.sources import ref
    return ref(path)

def parameters(w,head):return w.selected()+list(head.parameters())
def vector(w,head):return torch.cat([p.detach().flatten().clone() for p in parameters(w,head)])
def gradient(loss,params):
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,grads)])
def group_norm(v,split):return dict(motion=float(v[:split].norm()),head=float(v[split:].norm()))

def cache_data(cfg,w,d,data):
    pre,rows=population(cfg);parity=np.load(checked(pre["parity"]))
    parents=list(map(int,parity["parents"]));offsets=torch.tensor(parity["offsets"],device="cuda")
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda");cache={};errors=[]
    for row in rows:
        z,y,m=tensors(row);origin=torch.tensor(z["origin"][None],device="cuda")
        with torch.no_grad():error=float(abs(fk(y,scale,bias,origin,parents,offsets)[0].cpu().numpy()-parity["fit_"+str(row["case"])]).max())
        assert error<=1e-5;errors.append(error);cache[row["case"]]=(z,y,m,origin)
    return dict(rows=rows,cache=cache,parents=parents,offsets=offsets,scale=scale,bias=bias,parity=max(errors),diffusion=d)

def state(cfg,cache,case,timestep,hard,seed):
    z,y,m,origin=cache["cache"][case]
    v,k,h=observed(y,"inbetween",hard=hard)
    meta=torch.tensor(metadata()[None],device="cuda")
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);t=torch.tensor([timestep],device="cuda")
    xt=cache["diffusion"].q_sample(y,t)
    return y,m,origin,v,k,h,meta,t,xt

def schedule(cfg):
    rng=np.random.default_rng(cfg["seed"]);items=[]
    for block in range(cfg["steps"]//8):
        pairs=[(c,t) for c in cfg["cases"] for t in cfg["timesteps"]]
        for index in rng.permutation(len(pairs)):
            case,t=pairs[index];step=len(items)
            items.append(dict(step=step,case=case,t=t,hard=block%2==0,seed=cfg["seed"]+10000+step))
    assert len(items)==cfg["steps"];return items

def coverage(known,mode):
    masks=torch.cat((known[:,:150,4:],known[:,75:,4:]),0)
    clock=torch.stack((torch.arange(150,device=known.device),torch.arange(75,225,device=known.device))).float()
    index,positions,allowed=memory_support(clock,mode);masks=masks[index]
    return [dict(admitted=int(a.sum()),observed_fraction=float(m[a].float().mean()),
        fully_observed_keys=int(m[a].all(-1).sum()),unobserved_keys=int((~m[a].any(-1)).sum())) for m,a in zip(masks,allowed)]

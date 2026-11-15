import numpy as np
import torch
from pathlib import Path
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.branch_r22.lodge import load
from .model import ReferenceAdapter,digest
from .reference import view

def runtime(cfg):
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    d,n,s,f,e,info,manifest=load(read("src/dunhuang/research/branch_r22/config.json"))
    torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    a=ReferenceAdapter(False,cfg["width"],cfg["part_width"],cfg["injection_layers"],d.model.latent_dim).cuda()
    assert not any(p.requires_grad for p in d.model.parameters())
    return d,n,s,f,e,info,a

def dataset(root):
    root=Path(root);result=read(root/"data/result.json")
    for name,h in read(root/"data/protocol.json")["files"].items():assert sha(root/"data"/name)==h
    return result,{r["case"]:dict(np.load(root/"data"/r["arrays"])) for r in result["records"]}

def packet(rotations,seeds=None,augment=False,drops=None,control="correct"):
    rows=[view(r,None if seeds is None else seeds[i],augment,False if drops is None else drops[i],control) for i,r in enumerate(rotations)]
    return {k:torch.tensor(np.stack([r[k] for r in rows]),device="cuda") for k in rows[0]}

def install(adapter,p):adapter.set_reference(**p)

def ledger(cfg,cases):
    rng=np.random.default_rng(cfg["seed"]);order=[]
    while len(order)<cfg["steps"]*2:order.extend(np.array(cases)[rng.permutation(len(cases))].tolist())
    return [dict(step=i,cases=order[2*i:2*i+2],t=int(rng.integers(0,1000)),seed=cfg["seed"]+10000+i,
        reference_drop=[False,i%cfg["reference_dropout_second_row_every"]==0],music_dropout_probability=cfg["music_dropout"]) for i in range(cfg["steps"])]

def batch(cfg,data,row):
    torch.manual_seed(row["seed"]);torch.cuda.manual_seed_all(row["seed"])
    y=torch.tensor(np.stack([data[i]["target"] for i in row["cases"]]),device="cuda")
    music=torch.tensor(np.stack([data[i]["music"] for i in row["cases"]]),device="cuda")
    ref=packet([data[i]["raw139"][64:160,7:] for i in row["cases"]],[row["seed"],row["seed"]+1],True,row["reference_drop"])
    return y,music,ref

def objective(d,adapter,y,music,reference,t,seed,cfg,watch=False):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    ts=torch.full((len(y),),t,device=y.device,dtype=torch.long);noise=torch.randn_like(y);xt=d.q_sample(y,ts,noise)
    adapter.clear()
    # Match music dropout exactly for the frozen teacher and adapted forward.
    rng=torch.cuda.get_rng_state();cpu=torch.get_rng_state()
    with torch.no_grad():teacher=d.model(xt,music,None,ts,cond_drop_prob=cfg["music_dropout"])
    torch.cuda.set_rng_state(rng);torch.set_rng_state(cpu);install(adapter,reference);adapter.capture=watch
    pred=d.model(xt,music,None,ts,cond_drop_prob=cfg["music_dropout"]);adapter.clear()
    a,b=cfg["core"];parts=dict(core_body=(pred[:,a:b,7:]-y[:,a:b,7:]).square().mean(),
        core_velocity=(torch.diff(pred[:,a:b,7:],dim=1)-torch.diff(y[:,a:b,7:],dim=1)).square().mean(),
        root_preservation=(pred[:,:,4:7]-teacher[:,:,4:7]).square().mean(),
        outside_preservation=torch.cat((pred[:,:a,7:]-teacher[:,:a,7:],pred[:,b:,7:]-teacher[:,b:,7:]),1).square().mean())
    loss=sum(cfg["loss_weights"][k]*v for k,v in parts.items())
    return loss,pred,{k:float(v.detach()) for k,v in parts.items()},dict(adapter.readouts)

def checkpoint(adapter,path,cfg,arm,step):
    torch.save(dict(state={n:p.detach().cpu().clone() for n,p in adapter.state_dict().items()},config=cfg,arm=arm,step=step),path)

def restore(adapter,path,cfg,arm):
    z=torch.load(path,map_location="cpu",weights_only=True);assert z["config"]==cfg and z["arm"]==arm
    adapter.load_state_dict(z["state"],strict=True);adapter.phase=arm=="phase"

def vector(adapter):return torch.cat([p.detach().flatten() for p in adapter.parameters()]).clone()

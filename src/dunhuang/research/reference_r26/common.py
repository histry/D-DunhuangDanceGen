"""Shared R26 data, complete/occluded reference ledger and paired geometry targets."""
from pathlib import Path
import math
import numpy as np
import torch
from dunhuang.research.lodge_adapter_r25.common import dataset,checkpoint,restore,vector,read,fresh,commit,forbid_optimizers,same_arrays
from dunhuang.research.branch_r22.lodge import load
from .model import Adapter,digest
from .reference import view
from .geometry import fk22,relations,supported_losses

def runtime(cfg,arm):
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    d,n,s,f,e,info,_=load(read("src/dunhuang/research/branch_r22/config.json"))
    torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    a=Adapter(arm!="C0",width=64,part_width=16,layers=(1,4,7),hidden=d.model.latent_dim).cuda()
    return d,n,s,f,e,info,a

def packet(rotations,seeds=None,occluded=None,drops=None):
    rows=[view(r,None if seeds is None else seeds[i],False if occluded is None else occluded[i],False if drops is None else drops[i]) for i,r in enumerate(rotations)]
    return {k:torch.tensor(np.stack([r[k] for r in rows]),device="cuda") for k in rows[0]}

def install(adapter,reference):
    adapter.set_reference(**{k:v for k,v in reference.items() if k not in ("geometry_target","geometry_valid")})

def ledger(cfg,cases):
    rng=np.random.default_rng(cfg["seed"]);order=[]
    while len(order)<cfg["steps"]*2:order.extend(np.array(cases)[rng.permutation(len(cases))].tolist())
    return [dict(step=i,cases=order[2*i:2*i+2],t=int(rng.integers(0,1000)),seed=cfg["seed"]+10000+i,
        reference_drop=[False,i%5==0],occluded=[i%2==1,i%2==0]) for i in range(cfg["steps"])]

def batch(cfg,data,row):
    torch.manual_seed(row["seed"]);torch.cuda.manual_seed_all(row["seed"])
    y=torch.tensor(np.stack([data[c]["target"] for c in row["cases"]]),device="cuda")
    music=torch.tensor(np.stack([data[c]["music"] for c in row["cases"]]),device="cuda")
    p=packet([data[c]["raw139"][64:160,7:] for c in row["cases"]],[row["seed"],row["seed"]+1],row["occluded"],row["reference_drop"])
    for field in ("geometry_target","geometry_valid"):
        p[field]=torch.tensor(np.stack([data[c][field] for c in row["cases"]]),device="cuda")
    return y,music,p

def objective(d,adapter,y,music,reference,t,seed,cfg,arm,n,s,watch=False):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    ts=torch.full((len(y),),t,device=y.device,dtype=torch.long)
    xt=d.q_sample(y,ts,torch.randn_like(y));adapter.clear()
    state=torch.cuda.get_rng_state();cpu=torch.get_rng_state()
    with torch.no_grad():teacher=d.model(xt,music,None,ts,cond_drop_prob=.2)
    torch.cuda.set_rng_state(state);torch.set_rng_state(cpu);install(adapter,reference);adapter.capture=watch
    pred=d.model(xt,music,None,ts,cond_drop_prob=.2);layers=dict(adapter.readouts);adapter.clear()
    a,b=cfg["core"]
    parts=dict(core_body=(pred[:,a:b,7:]-y[:,a:b,7:]).square().mean(),
        core_velocity=(pred[:,a+1:b,7:]-pred[:,a:b-1,7:]-y[:,a+1:b,7:]+y[:,a:b-1,7:]).square().mean(),
        root_preservation=(pred[:,:,4:7]-teacher[:,:,4:7]).square().mean(),
        outside_preservation=torch.cat((pred[:,:a,7:]-teacher[:,:a,7:],pred[:,b:,7:]-teacher[:,b:,7:]),1).square().mean())
    weights=dict(core_body=1.,core_velocity=.1,root_preservation=.1,outside_preservation=.25)
    loss=sum(weights[k]*v for k,v in parts.items())
    if arm=="C2":
        points=fk22(n.unnormalize(pred[:,a:b].clone()),s)
        pos,vel=supported_losses(relations(points),reference["geometry_target"][:,a:b],reference["geometry_valid"][:,a:b])
        parts.update(geometry=pos,geometry_velocity=vel);loss=loss+cfg["geometry_weight"]*pos+cfg["geometry_velocity_weight"]*vel
    return loss,pred,{k:float(v.detach()) for k,v in parts.items()},layers

def gradients(adapter):
    prefixes=dict(encoder=("joint","parts"),sequence=("temporal",),input=("input",),projections=("out",))
    return {k:float(sum((p.grad.detach().square().sum() for name,p in adapter.named_parameters() if name.startswith(v) and p.grad is not None),torch.tensor(0.,device="cuda")).sqrt()) for k,v in prefixes.items()}

def check_readout(actual,expected):
    assert {k:v for k,v in actual.items() if k!="gradient_norms"}=={k:v for k,v in expected.items() if k!="gradient_norms"}
    deltas={}
    assert actual["gradient_norms"].keys()==expected["gradient_norms"].keys()
    for k,x in actual["gradient_norms"].items():
        y=expected["gradient_norms"][k];assert math.isfinite(x) and math.isfinite(y) and min(x,y)>=0
        assert math.isclose(x,y,rel_tol=1e-5,abs_tol=0),k
        deltas[k]=dict(actual=x,expected=y,relative_error=abs(x-y)/max(x,y) if x or y else 0.)
    return dict(non_gradient_fields_exact=True,gradient_norm_rtol=1e-5,gradient_norm_atol=0.,deltas=deltas)

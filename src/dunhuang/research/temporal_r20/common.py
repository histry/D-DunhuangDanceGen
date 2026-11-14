"""Sealed warm start, true joint225 and split150 layouts, fixed Train/DEV roles."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from dunhuang.io import checked,sha
from dunhuang.research.source_capacity.common import read
from dunhuang.research.objective_r19.common import load as old_load,cache_data as old_cache,source_config as older_config,restore
from dunhuang.research.task_denoising_r18.engine import population as old_population
from dunhuang.research.global_clock_r17.engine import tensors
from dunhuang.research.global_clock_r17.model import fk
from .trace import TracedPredictor
from .joint import JointDenoiser,JointState,JointPredictor

def parent_config(cfg):
    root=Path(cfg["parent"]);seal=read("reports/r19_receipt.json")
    for n in ("calibration/protocol.json","calibration/result.json","models/N2/result.json"):
        assert sha(root/n)==seal["evidence"][n],n
    return read(root/"calibration/protocol.json")["config"]

def load(cfg,layout="split",reference=None):
    pc=parent_config(cfg);warm=read(Path(cfg["parent"])/"models/N2/result.json")["checkpoint"]
    w,d,data,p,h=old_load(pc,"self",warm)
    if layout=="joint":
        p.close();w=JointDenoiser(w);newhead=JointState().cuda();newhead.load_state_dict(h.state_dict());h=newhead;p=JointPredictor(w,h)
    elif layout!="split":raise ValueError("Explicit split/joint layout required")
    if reference:restore(cfg,w,h,reference)
    return w,d,data,TracedPredictor(p),h

def selected_digest(w):
    h=hashlib.sha256()
    for n,p in sorted(w.state().items()):h.update(n.encode());h.update(p.numpy().tobytes())
    return h.hexdigest()

def frozen_except_null(w):
    h=hashlib.sha256()
    for n,p in sorted(w.net.named_parameters()):
        if p.requires_grad or n=="null_cond_embed":continue
        h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def cache_data(cfg,w,d,data):
    pc=parent_config(cfg);cache=old_cache(pc,w,d,data)
    pre,fit,ev=old_population(older_config(pc));parity=np.load(checked(pre["parity"]))
    rows=[r for r in fit if r["case"] in cfg["train_cases"]+cfg["dev_cases"]]
    assert len(rows)==4
    for row in rows:
        case=row["case"]
        if case in cache["cache"]:continue
        z,y,m=tensors(row);origin=torch.tensor(z["origin"][None],device="cuda")
        with torch.no_grad():err=float(abs(fk(y,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])[0].cpu().numpy()-parity["fit_"+str(case)]).max())
        assert err<=1e-5;cache["cache"][case]=(z,y,m,origin)
    cache["rows"]=rows;cache["assets"]=pre["assets"]
    assert not {r["left"]["source_group"] for r in rows if r["case"] in cfg["train_cases"]}&{r["left"]["source_group"] for r in rows if r["case"] in cfg["dev_cases"]}
    return cache

def grid():return list(reversed(torch.linspace(-1,999,51).int().tolist()))[:-1]

def schedule(cfg,arm):
    from dunhuang.research.objective_r19.common import schedule as old
    rows=old(dict(seed=cfg["seed"],steps=cfg["steps"],cases=cfg["train_cases"],timesteps=[999,519,119,19]))
    rng=np.random.default_rng(cfg["seed"]+70);g=grid();shuffled=[]
    while len(shuffled)<cfg["steps"]//2:shuffled.extend(np.array(g)[rng.permutation(50)].tolist())
    for i,row in enumerate(rows):
        row["slot"]="grid" if i%2 else "uniform"
        if arm not in ("S0","S1","A0"):
            row["t"]=int(shuffled[i//2]) if i%2 else int(rng.integers((i//2%10)*100,(i//2%10+1)*100))
        row["use_cache"]=arm=="A3" and row["slot"]=="grid"
    return rows

import hashlib
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha
from dunhuang.research.source_capacity.common import read
from dunhuang.research.motion_bridge.study import population
from dunhuang.research.boundary_context.common import Data,load_model,masks
from .model import MaskConditionedDenoiser

def assets_and_population(cfg):
    r12=Path(cfg["r12_root"]);seal=read("reports/r12_summary.json")
    for name in ("capacity/result.json","cache/result.json","cache/protocol.json","B3/result.json"):
        if sha(r12/name)!=seal["evidence"][name]:raise ValueError("Frozen R12 seal differs: "+name)
    assets,train,groups,select=population(read(cfg["r12_config"]))
    cache=read(r12/"cache/result.json");fits=[b for b in cache["bindings"] if b["role"]=="fit"]
    monitors=[b for b in cache["bindings"] if b["role"]=="monitor"]
    if len(fits)!=8 or len(monitors)!=4:raise ValueError("Expected registered R12 populations")
    rows=read(r12/"cache/protocol.json")["fit_rows"]
    if set(r["source_group"] for r in rows)&set(b["row"]["source_group"] for b in monitors):raise ValueError("Train monitor leakage")
    return assets,rows,fits[:cfg["tiny_windows"]],monitors,fits

def setup(cfg):
    assets,rows,tiny,monitors,fits=assets_and_population(cfg)
    data=Data(np.load(checked(assets["scaler"])));model,diffusion,_=load_model(assets)
    return assets,rows,tiny,monitors,fits,data,model,diffusion

def make(cfg,model):
    return MaskConditionedDenoiser(model,cfg["trainable_prefixes"]).cuda()

def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def checkpoint_load(cfg,wrapper,reference,assets):
    p=torch.load(checked(reference),map_location="cpu",weights_only=True)
    if p["schema"]!=cfg["schema"] or p["config"]!=cfg or p["base_sha256"]!=assets["checkpoint"]["sha256"]:
        raise ValueError("Masked denoiser lineage differs")
    wrapper.load_selected(p["state"]);return p

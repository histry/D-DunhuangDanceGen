"""R13 selected layers adapt; frozen music and unselected weights remain sealed."""
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked
from dunhuang.research.source_capacity.common import read
from dunhuang.research.global_clock_r17.engine import load as parent_load
from dunhuang.research.global_clock_r17.model import Predictor as OldPredictor
from .model import State,Predictor

def parent_config(cfg):return read(Path(cfg["parent"])/"data/protocol.json")["config"]

def population(cfg):
    root=Path(cfg["parent"]);seal=read("reports/r17_receipt.json")
    from dunhuang.io import sha
    for name in ("data/protocol.json","data/result.json"):
        assert sha(root/name)==seal["evidence"][name],name
    pre=read(root/"data/protocol.json");rows=read(root/"data/result.json")["records"]
    fit=[r for r in rows if r["role"]=="fit"];ev=[r for r in rows if r["role"]=="eval"]
    assert len(fit)==8 and len(ev)==4
    assert not {r["left"]["source_group"] for r in fit}&{r["left"]["source_group"] for r in ev}
    for r in rows:checked(r["arrays"])
    checked(pre["parity"]);return pre,fit,ev

def load(cfg,mode=None,reference=None,adapt=True):
    wrapper,diffusion,data,old,_=parent_load(parent_config(cfg));old.close()
    # The original R13 names are the explicit allowed motion prefixes.
    for name,p in wrapper.net.named_parameters():p.requires_grad_(adapt and name in wrapper.trainable_names)
    head=None
    if mode:
        torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
        head=State(mode,cfg["state_width"]).cuda()
    if reference:
        state=torch.load(checked(reference),map_location="cpu",weights_only=True)
        assert state["config"]==cfg and state["mode"]==mode
        wrapper.load_selected(state["motion"]);head.load_state_dict(state["head"])
    wrapper.eval()
    return wrapper,diffusion,data,Predictor(wrapper,head),head

def unknown_mse(pred,y,known):
    mask=~known[:,:,4:]
    return (pred[:,:,4:]-y[:,:,4:]).square()[mask].mean()

def trainable(wrapper,head):return wrapper.selected()+list(head.parameters())

def supervision(step):
    task=("free","inbetween","progressive")[step%3]
    # Each task receives exactly20 ranking updates in a120-update budget.
    return task,step%2==0

def amplitude(pred,target,scale,bias,origin,parents,offsets):
    from dunhuang.research.global_clock_r17.model import fk
    p=fk(pred,scale,bias,origin,parents,offsets);g=fk(target,scale,bias,origin,parents,offsets)
    pv=torch.diff(p,dim=1)*30;gv=torch.diff(g,dim=1)*30
    # Match per-joint speed magnitude and temporal excursion; no static-foot labels.
    speed=(pv.norm(dim=-1).mean(1)-gv.norm(dim=-1).mean(1)).square().mean()
    extent=((p-p.mean(1,keepdim=True)).square().mean(1).clamp_min(1e-8).sqrt()-
            (g-g.mean(1,keepdim=True)).square().mean(1).clamp_min(1e-8).sqrt()).square().mean()
    return speed+extent

def history_contract(total=225,committed=59,buffer_end=75):
    if not 0<committed<buffer_end<total:raise ValueError("History/buffer/future ordering")
    return dict(committed=[0,committed],editable_buffer=[committed,buffer_end],
        future=[buffer_end,total],generated_history_training=False,qualified=False)

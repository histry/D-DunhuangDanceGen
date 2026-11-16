"""Pinned R33 runtime and source-only full-length packets; all unknown Root remains unknown."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from dunhuang.research.control_r33.common import *
from dunhuang.research.control_r33.runtime import Runtime as ParentRuntime
from dunhuang.research.reference_r26.reference import features
from dunhuang.research.contract_r28.representation import rotations
from .model import WindowReference

@dataclass
class WindowSource:
    values:dict
    source_uids:tuple
    def branch(self):return "change"
    def validate(self):
        x=self.values["features"];valid=self.values["valid"]
        if x.ndim!=4 or x.shape[2:]!=(22,9) or x.shape[1]<2 or valid.shape!=x.shape[:-1] or valid.dtype!=torch.bool:raise ValueError("Explicit-length observed body packet required")
        if len(self.source_uids)!=len(x) or not all(self.source_uids) or not torch.isfinite(x[valid]).all():raise ValueError("Source identities/finite observations required")
        return self

class Runtime:
    def __init__(self,cfg,checkpoint=None,context_training=False):
        p=Path(cfg["r33"]);assert sha(p/"artifacts_manifest.json")==cfg["r33_manifest_sha256"]
        start=p/"models/Dyn/adapter.pt";assert sha(start)==cfg["dyn_checkpoint_sha256"]
        base=ParentRuntime(read("src/dunhuang/research/control_r33/config.json"),start)
        for name in ("d","n","s","f","e","info","io","c","old"):setattr(self,name,getattr(base,name))
        reference=WindowReference(4.).cuda();reference.load_state_dict(self.c.change.state_dict(),strict=True);self.c.change=reference
        self.context_training=context_training
        self.c.change.requires_grad_(not context_training);self.c.context_branch.requires_grad_(context_training)
        if checkpoint:self.c.change.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["state"],strict=True)
        self.frozen=self.fingerprint()
    def fingerprint(self):
        result=dict(backbone=digest(self.d.model),io=digest(self.io),general=digest(self.c.general))
        key="change" if self.context_training else "context_branch";result[key]=digest(getattr(self.c,key))
        return result
    def check(self):
        assert self.fingerprint()==self.frozen
        key="change" if self.context_training else "context_branch"
        assert all(p.grad is None for m in (self.d.model,self.io,self.c.general,getattr(self.c,key)) for p in m.parameters())
    def close(self):self.check();self.c.close();self.old.close()

def save(rt,path,cfg,arm,step):
    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in rt.c.change.state_dict().items()},config=cfg,arm=arm,step=step,frozen=rt.frozen),path)
def restore(rt,path):
    x=torch.load(path,map_location="cpu",weights_only=True);assert x["frozen"]==rt.frozen;rt.c.change.load_state_dict(x["state"],strict=True)

def full_packet(body,uids,mode="correct",drops=None,halo=None):
    body=np.asarray(body,np.float32)
    if body.ndim!=3 or body.shape[1:]!=(256,132):raise ValueError("Actual256 body observations required")
    order=list(range(len(body)))
    if mode=="swapped":order=list(range(1,len(body)))+[0]
    elif mode not in ("correct","wrong_clock"):raise ValueError("Reference mode")
    fs=[]
    for j in order:
        r=rotations(body[j])
        if halo is None:f,_=features(r)
        else:f,_=features(rotations(np.concatenate((halo[j:j+1],body[j]))));f=f[1:]
        fs.append(f)
    valid=np.ones((len(body),256,22),bool)
    if drops is not None:
        for i,drop in enumerate(drops):
            if drop:valid[i]=False
    t=np.arange(256)/255.;clock=np.stack((np.sin(2*np.pi*t),np.cos(2*np.pi*t)),-1).astype(np.float32)
    mapped=np.broadcast_to(np.arange(256,dtype=np.float32),(len(body),256)).copy()
    if mode=="wrong_clock":mapped=np.minimum(mapped+24,255)
    values=dict(features=torch.tensor(np.stack(fs),device="cuda"),valid=torch.tensor(valid,device="cuda"),
        clock=torch.tensor(np.broadcast_to(clock,(len(body),256,2)).copy(),device="cuda"),
        ref_phase=torch.zeros(len(body),256,8,device="cuda"),query_phase=torch.zeros(len(body),256,8,device="cuda"),
        mapped_source=torch.tensor(mapped,device="cuda"),control_mask=torch.ones(len(body),256,device="cuda",dtype=torch.bool))
    return WindowSource(values,tuple(uids[j] for j in order))

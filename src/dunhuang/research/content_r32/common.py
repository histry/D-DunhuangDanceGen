"""R32 binds the complete R31 platform; only reference parameters may learn."""
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.geometry_r31.common import sources as old_sources
from dunhuang.research.geometry_r31.interface import runtime as old_runtime
from dunhuang.research.state_r27.common import read,sha,checked,digest,restore,metrics
from dunhuang.research.contract_r28.representation import packet as make_packet
from dunhuang.research.time_r29.model import mapping

def verify_parent(cfg):
    root=Path(cfg["r31"]);assert sha(root/"artifacts_manifest.json")==cfg["r31_manifest_sha256"]
    path=root/"models/CanonicalIO/adapter.pt";assert sha(path)==cfg["canonical_checkpoint_sha256"]
    return path

def population(cfg):
    verify_parent(cfg)
    rows,ledger,old=old_sources(read("src/dunhuang/research/geometry_r31/config.json"))
    aist={r["case"]:r["data"] for r in rows if r["domain"]=="aist"}
    originals=np.load(checked(read("outputs/lodge_reference_r24_01/main/protocol.json")["arrays"]))
    return rows,ledger,aist,originals

def runtime(cfg,checkpoint=None):
    start=verify_parent(cfg)
    d,n,s,f,e,info,b=old_runtime(read("src/dunhuang/research/geometry_r31/config.json"),True,start)
    if checkpoint:restore(b,checkpoint)
    b.io.requires_grad_(False)
    assert sum(p.numel() for p in b.parameters() if p.requires_grad)==265328
    assert all(not p.requires_grad for p in d.model.parameters())
    return d,n,s,f,e,info,b

def core_body(row):
    x=row["model_raw"]
    return x[64:160,7:] if len(x)==256 else x[:,7:]

def core_points(row):
    p=row["model_points"]
    return p[64:160] if len(p)==256 else p

def reference(bodies,mode="correct",drops=None):
    donor=list(bodies[1:])+list(bodies[:1]) if mode=="swapped" else bodies
    p=make_packet(donor,drops=drops)
    p["mapped_source"]=mapping(len(bodies),p["features"].device,wrong=mode=="wrong_clock")
    return p

def local_metrics(body,truth,points,s,f):
    """Zero translation is ONLY a FK gauge for 96-frame local diagnostic metrics."""
    x=torch.zeros(1,96,139,device="cuda");x[0,:,7:]=torch.as_tensor(body,device="cuda").clamp(-1,1)
    with torch.no_grad():p=f(x,s)[0,:,:22].cpu().numpy()
    target=np.zeros((96,139),np.float32);target[:,7:]=truth
    return metrics(x[0].cpu().numpy(),p,target,points,mask=np.ones(96,bool))

def fingerprint(d,b):
    return dict(backbone=digest(d.model),action_io=digest(b.io))

def assert_frozen(d,b,expected):
    assert fingerprint(d,b)==expected
    assert all(p.grad is None for p in d.model.parameters()) and all(p.grad is None for p in b.io.parameters())

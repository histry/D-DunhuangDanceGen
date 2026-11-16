"""Shared pinned loaders and semantic packets; no frozen baseline source is modified."""
from pathlib import Path
import numpy as np
from dunhuang.research.state_r27.common import runtime as old_runtime
from dunhuang.research.state_r27.common import read,sha,restore
from dunhuang.research.contract_r28.representation import packet as canonical_packet
from dunhuang.research.contract_r28.train import dataset
from .model import TimeReference,mapping

def runtime(cfg,beta=4.,checkpoint=None):
    d,n,s,f,e,info,a=old_runtime(read("src/dunhuang/research/state_r27/config.json"))
    state=a.state_dict();a.close();new=TimeReference(beta).cuda();new.load_state_dict(state,strict=True)
    if checkpoint is not None:restore(new,checkpoint)
    new.attach(d.model);return d,n,s,f,e,info,new

def packet(rows,mode="correct"):
    source=rows[1:]+rows[:1] if mode=="swapped" else rows
    p=canonical_packet([x["model_raw"][64:160,7:] for x in source])
    p["mapped_source"]=mapping(len(rows),p["features"].device,wrong=mode=="wrong_clock")
    return p

def rows(cfg,root,domain,role):
    if domain=="aist":
        cases=cfg["train_cases"] if role=="train" else cfg["holdout_cases"]
        return dataset(cfg["r28"],cases),cases
    result=read(Path(root)/"fine_data/result.json")
    selected=[r for r in result["records"] if r["role"]==role]
    values=[]
    for r in selected:
        p=Path(root)/"fine_data"/r["arrays"];assert sha(p)==r["sha256"];values.append(dict(np.load(p)))
    return values,[r["item"] for r in selected]

"""Sealed R29 inputs, fixed source population and exact scientific lineage."""
from pathlib import Path
import numpy as np
from dunhuang.research.state_r27.common import read,sha,restore
from dunhuang.research.time_r29.common import runtime as previous_runtime,rows,packet
from .model import Reference

def parent(cfg):
    p=Path(cfg["r29"])
    assert sha(p/"artifacts_manifest.json")==cfg["r29_manifest_sha256"]
    return p

def checked_parent(cfg,relative):
    p=parent(cfg);entry=read(p/"artifacts_manifest.json")["files"][relative]
    assert sha(p/relative)==entry["sha256"]
    return p/relative

def runtime(cfg,aligned=False,checkpoint=None):
    d,n,s,f,e,info,old=previous_runtime(read("src/dunhuang/research/time_r29/config.json"))
    state=old.state_dict();old.close();a=Reference(aligned).cuda();a.load_state_dict(state,strict=True);a.attach(d.model)
    if checkpoint:restore(a,checkpoint)
    return d,n,s,f,e,info,a

def values(cfg,domain,role):
    return rows(read("src/dunhuang/research/time_r29/config.json"),parent(cfg),domain,role)

def cohort(cfg):
    p=parent(cfg);pop=read(checked_parent(cfg,"coverage_data/result.json"))
    data={}
    for row in pop["records"]:
        path=checked_parent(cfg,"coverage_data/"+row["arrays"]);assert sha(path)==row["sha256"]
        data[row["case"]]=dict(np.load(path))
    original=read(checked_parent(cfg,"models/COVER/result.json"))
    ledger=[{k:row[k] for k in ("step","cases","seed","t")} for row in original["history"]]
    return data,ledger,original

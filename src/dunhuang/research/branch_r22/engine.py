"""R21 B3 warm values, R22 snapshots, and one identical ledger across arms."""
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha
from dunhuang.research.source_capacity.common import read
from dunhuang.research.context_r21 import engine as old
from dunhuang.research.objective_r19.common import restore
from dunhuang.research.objective_r19.objective import reconstruction
from dunhuang.research.temporal_r20.common import grid
from .objective import branch_scores

model_meta=old.model_meta
cache_data=old.cache_data
state=old.state
weights=old.weights

def parent(cfg):
    p=Path(cfg["parent"]);seal=read("reports/r21_release/r21_receipt.json")
    for name in ("data/result.json","data/protocol.json","models/B3/result.json","models/B3/protocol.json"):
        assert sha(p/name)==seal["evidence"][name],("Unsealed R21 input",name)
    return read(p/"data/protocol.json")["config"]

def load(cfg,root,arm,reference=None):
    if arm not in cfg["arms"]:raise ValueError("Unregistered arm")
    pc=parent(cfg);warm=read(Path(cfg["parent"])/"models/B3/result.json")
    w,d,data,p,h=old.load(pc,cfg["parent"],"B3",warm["checkpoint"])
    assert old.model_meta(w,h)["selected"]==warm["final_selected"]
    assert old.model_meta(w,h)["head"]==warm["final_head"]
    if reference:restore(cfg,w,h,reference)
    return w,d,data,p,h

def schedule(cfg,population):
    return old.schedule(cfg,population,"B3")

def music_state(cache,s,case,mode):
    if mode=="correct":return s,None
    out=list(s)
    if mode=="music_zero":
        out[1]=torch.zeros_like(s[1]);return out,dict(mode="feature_zero_ablation")
    if mode!="music_swapped":raise ValueError("Unknown music control")
    cases=cache["population"]["train_probe"] if case in cache["population"]["train_cases"] else cache["population"]["dev_cases"]
    donor=cases[(cases.index(case)+1)%len(cases)]
    out[1]=cache["cache"][donor][2]
    assert not torch.equal(out[1],s[1])
    return out,dict(donor=donor,source=next(r for r in cache["rows"] if r["case"]==donor)["audio225"])

def control_cases(cache):
    return cache["population"]["train_probe"][:2]+cache["population"]["dev_cases"][:2]

def probes(cfg,cache,predict,controls=False,dev=False):
    cases=cache["population"]["train_probe"]+(cache["population"]["dev_cases"] if dev else [])
    scores=[];arrays={}
    with torch.no_grad():
        for case in cases:
            for variant in cfg["eval_variants"] if dev else [0]:
                for t in grid() if variant==0 else [999,519,119,19]:
                    s,k,_=state(cfg,cache,case,t,variant=variant)
                    predict.capture_branches=True
                    pred,_=predict(s[-1],s[1],s[3],s[4],s[6],s[7])
                    terms,counts=reconstruction(pred,s[0],k)
                    row=dict(case=case,variant=variant,t=t,unknown=float(terms["unknown"]),observed=float(terms["observed"]),counts=counts,
                        branches=branch_scores(predict,s[0],k))
                    if t in (999,519,119,19):
                        stem=f"{case}_{variant}_{t}"
                        arrays[stem]=pred[0].cpu().numpy()
                        for name,x in zip(("unc_raw","cond_raw"),predict.branches):arrays[stem+"_"+name]=x[0].cpu().numpy()
                    if controls and t in (999,519,119,19) and case in control_cases(cache) and variant==0:
                        row["controls"]={}
                        for mode in ("music_swapped","music_zero"):
                            cs,_=music_state(cache,s,case,mode)
                            cp,_=predict(cs[-1],cs[1],cs[3],cs[4],cs[6],cs[7])
                            row["controls"][mode]=float(reconstruction(cp,s[0],k)[0]["unknown"])
                    scores.append(row)
    predict.capture_branches=False
    return scores,arrays

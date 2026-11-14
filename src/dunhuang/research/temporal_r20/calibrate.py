"""Calibrate added paired temporal terms at the selected common structural endpoint."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.objective_r19.common import state,parameters,gradient,group_norm
from dunhuang.research.objective_r19.objective import reconstruction
from .common import load,cache_data
from .objective import temporal

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();choice=read(root/"structure_gate.json")
    w,d,data,predict,head=load(cfg,choice["layout"],choice["checkpoint"]);cache=cache_data(cfg,w,d,data)
    split=sum(p.numel() for p in w.selected());ps=parameters(w,head);records=[];arrays={}
    before=w.digest(True)
    for case in cfg["train_cases"]:
        for t in cfg["calibration_times"]:
            for hard in (False,True):
                s=state(cfg,cache,case,t,hard,cfg["seed"]+40000+case*1000+t)
                y,m,o,v,k,h,meta,ts,xt=s;p,_=predict(xt,m,v,k,meta,ts)
                base,_=reconstruction(p,y,k);extra,counts=temporal(p,y,v,k,h,cache,o,cfg["temporal_scales"])
                norms={n:group_norm(gradient(z,ps),split) for n,z in dict(unknown=base["unknown"],**extra).items()}
                records.append(dict(case=case,t=t,hard=hard,norms=norms,counts=counts))
                arrays[f"{case}_{t}_{hard}"]=np.array([[norms[n][g] for g in ("motion","head")] for n in ("unknown","body_velocity","body_acceleration")])
    reference={g:float(np.median([r["norms"]["unknown"][g] for r in records])) for g in ("motion","head")}
    weights={}
    for name in ("body_velocity","body_acceleration"):
        ratios=[float(np.median([r["norms"][name][g] for r in records]))/max(reference[g],1e-12) for g in reference]
        weights[name]=cfg["temporal_budget"]/(2*max(1.,*ratios))
    assert w.digest(True)==before and all(0<x<=.05 for x in weights.values())
    np.savez_compressed(out/"norms.npz",**arrays)
    result=dict(weights=weights,reference=reference,records=records,optimizer_steps=0,dev_used=False,operational_qualified=False)
    if replay:
        assert result==read(Path(replay)/"result.json");same_arrays(out/"norms.npz",Path(replay)/"norms.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,selected_structure=ref(root/"structure_gate.json")))
    predict.close();print("R20_CALIBRATION",weights,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

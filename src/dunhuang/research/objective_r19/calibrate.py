"""Fixed Train16-state gradient calibration, bounded static weights, zero updates."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from .common import load,cache_data,state,parameters,gradient,group_norm
from .objective import PHYSICS,reconstruction,physical,geometry,static_weights

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();w,d,data,predict,head=load(cfg);cache=cache_data(cfg,w,d,data)
    before=w.digest();params=parameters(w,head);split=sum(p.numel() for p in w.selected());records=[];arrays={}
    for case in cfg["cases"]:
        for t in cfg["timesteps"]:
            for hard in (False,True):
                s=state(cfg,cache,case,t,hard,cfg["seed"]+40000+case*100+t)
                y,m,o,v,k,h,meta,ts,xt=s;pred,_=predict(xt,m,v,k,meta,ts)
                r,counts=reconstruction(pred,y,k)
                phys,pc=physical(pred,y,v,k,h,cache["scale"],cache["bias"],o,cache["parents"],cache["offsets"],cfg["physics_scales"])
                norms={name:group_norm(gradient(value,params),split) for name,value in dict(unknown=r["unknown"],**phys).items()}
                key=f"{case}_{t}_{hard}";arrays[key]=np.array([[norms[name][g] for g in ("motion","head")] for name in ("unknown",*PHYSICS)])
                records.append(dict(case=case,t=t,hard=hard,norms=norms,counts=dict(counts,**pc),geometry=geometry(pred,cache["scale"],cache["bias"])))
    reference={g:float(np.median([r["norms"]["unknown"][g] for r in records])) for g in ("motion","head")}
    medians={name:{g:float(np.median([r["norms"][name][g] for r in records])) for g in reference} for name in PHYSICS}
    weights=static_weights(medians,reference,cfg["calibration_budget_ratio"],cfg["calibration_reference_floor"])
    assert all(np.isfinite(v) and 0<v<=cfg["calibration_budget_ratio"]/4 for v in weights.values())
    assert w.digest()==before
    np.savez_compressed(out/"norms.npz",**arrays)
    result=dict(weights=weights,reference=reference,median_physics_gradients=medians,records=records,
        optimizer_steps=0,fk_parity_max_m=cache["parity"],calibration_uses_eval=False,operational_qualified=False)
    if replay:
        assert result==read(Path(replay)/"result.json");same_arrays(out/"norms.npz",Path(replay)/"norms.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        parent=ref(Path(cfg["parent"])/"p0/protocol.json"),weight_upper_bound=cfg["calibration_budget_ratio"]/4,
        rule="budget/(4*max(1,median_motion_ratio,median_head_ratio)); frozen before any optimizer"))
    predict.close();print("R19_CALIBRATED",weights,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.out,a.replay_from)

"""Supplemental full-step SO3 endpoint readout; does not replace sealed P0 outputs."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write
from dunhuang.contracts import matrices
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data
from dunhuang.research.masked_denoiser.metrics import decode
from .common import population

def angle(R):return float(Rotation.from_matrix(R).magnitude())

def run(cfg,out,replay=None):
    out=fresh(out);assets,_,_,_,monitors=population(cfg);scaler=np.load(checked(assets["scaler"]));data=Data(scaler)
    records=[];inputs=[]
    for b in read(Path(cfg["r13_root"])/"main_rollout/result.json")["bindings"]:
        if b["driver"]!="correct":continue
        old=data.native(monitors[b["case"]]["row"]);G=matrices(old[:,7:13]);z=np.load(checked(b["trace"]));inputs.append(b["trace"])
        for step,p in enumerate(z["raw"]):
            x=decode(p,scaler,old);P=matrices(x[:,7:13]);gt=G[59].T@G[90];pred=P[59].T@P[90]
            records.append(dict(case=b["case"],arm=b["arm"],step=step,
                raw_left_orientation_error_rad=angle(G[59].T@P[59]),
                raw_right_orientation_error_rad=angle(G[90].T@P[90]),
                raw_relative_turn_error_rad=angle(gt.T@pred)))
    result=dict(records=records,optimizer_steps=0,new_neural_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,trace_inputs=inputs))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R14_TURN_AUDIT",len(records),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)

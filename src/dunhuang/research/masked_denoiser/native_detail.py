"""Supplement original-source and same-state teacher FK/Native diagnostics."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.contracts import MotionAdapter
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.boundary_context.common import Data
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.condition_dependency.codec import origin
from .common import assets_and_population
from .metrics import joins,physical,qmetrics

def run(cfg,root,out,replay=None):
    out=fresh(out);root=Path(root);assets,_,tiny,monitors,_=assets_and_population(cfg);native_environment(assets)
    scaler=np.load(checked(assets["scaler"]));data=Data(scaler);sources=[];probes=[]
    def save(folder,native,fk,audit):
        folder.mkdir();np.save(folder/"native.npy",native);np.save(folder/"fk.npy",fk);write(folder/"physical.json",audit)
        if replay:
            other=Path(replay)/folder.name
            assert np.array_equal(native,np.load(other/"native.npy")) and np.array_equal(fk,np.load(other/"fk.npy"))
            assert audit==read(other/"physical.json")
    for role,items in (("tiny",tiny),("monitor",monitors)):
        for i,b in enumerate(items):
            old=data.native(b["row"]);fk,audit=audit_native(old)
            sources.append(dict(role=role,case=i,row=b["row"],joins=joins(old,fk),native_ok=audit["absolute"]["ok"],reasons=audit["absolute"]["reasons"]))
            save(out/(role+"_"+str(i)),old,fk,audit)
    z=np.load(root/"teacher/probes.npz")
    for i,b in enumerate(tiny):
        old=data.native(b["row"]);target=MotionAdapter.to_edge(old,origin(old))*scaler["scale"]+scaler["bias"]
        for j,t in enumerate(cfg["timesteps"]):
            for control in ("B0","correct","wrong","random"):
                p=z[f"{j}_{i}_{control}"];x,fk,audit,score=physical(p,old,scaler)
                probes.append(dict(case=i,t=t,control=control,**score,raw_known=qmetrics(p,target,scaler)))
                save(out/(str(i)+"_"+str(t)+"_"+control),x,fk,audit)
    result=dict(sources=sources,teacher_same_state_probes=probes,optimizer_steps=0,new_neural_generations=0,
        original_native_thresholds_unchanged=True,probes_are_single_state_readouts_not_full_rollouts=True,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,teacher_source="teacher/probes.npz"))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R13_SOURCE_NATIVE",[(role,sum(s["native_ok"] for s in sources if s["role"]==role)) for role in ("tiny","monitor")],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

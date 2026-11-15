"""Regression only: new instrumentation must reproduce sealed R20 trajectories."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.temporal_r20.common import cache_data as old_cache
from dunhuang.research.objective_r19.common import state as old_state
from .engine import load,model_meta
from .sampler import sample

def run(cfg,root,out,replay=None):
    out=fresh(out);forbid_optimizers();w,d,data,p,h=load(cfg,root,"B0")
    pc=read(Path(cfg["r20_root"])/"interface/protocol.json")["config"];cache=old_cache(pc,w,d,data)
    original=read(Path(cfg["r20_root"])/"p0_rollout/result.json")["bindings"];before=model_meta(w,h);records=[]
    for case in (0,1):
        s=old_state(pc,cache,case,519,True,pc["seed"]+case)
        arrays,receipt=sample(cfg,cache,p,s,89041+case)
        old=next(b for b in original if b["case"]==case and b["arm"]=="WarmMean")
        z=np.load(checked(old["trace"]))
        for name in ("sample","raw_final","known","hard","condition"):assert np.array_equal(arrays[name],z[name])
        path=out/(str(case)+".npz");np.savez_compressed(path,**arrays)
        if replay:same_arrays(path,Path(replay)/path.name);assert receipt==read(Path(replay)/(str(case)+".json"))
        write(out/(str(case)+".json"),receipt);records.append(dict(case=case,exact_r20_sample=True,exact_r20_raw=True))
    assert model_meta(w,h)==before
    write(out/"result.json",dict(records=records,optimizer_steps=0))
    write(out/"protocol.json",dict(config=cfg,code_commit=commit(),purpose="sampler regression, no repeated joint visibility experiment"))
    p.close();print("R21_NEUTRALITY",records,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

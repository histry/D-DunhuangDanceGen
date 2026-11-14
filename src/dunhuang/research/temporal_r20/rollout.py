"""Structure side interventions precede DEV; final evaluation is frozen and seed matched."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.objective_r19.common import state
from .common import load,cache_data
from .conditions import endpoint,CONTROLS
from .sampler import sample

def run(cfg,root,out,part,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    if part=="p0":arms=["WarmMean","WarmQuintic"];cases=cfg["train_cases"];seeds=[0]
    elif part=="structure":arms=["S0","S1"];cases=cfg["train_cases"];seeds=[0]
    elif part=="gap":arms=["A2"];cases=cfg["train_cases"];seeds=[0]
    elif part=="final":
        arms=["S0","S1","A0","A1","A2"]+(["A3"] if read(root/"cache_gate.json")["cache_admitted"] else [])
        cases=cfg["train_cases"]+cfg["dev_cases"];seeds=cfg["eval_seeds"]
    else:raise ValueError("Unknown study part")
    bindings=[]
    for arm in arms:
        if arm.startswith("Warm"):layout="split";reference=None
        else:
            r=read(root/"models"/arm/"result.json");layout=r["layout"];reference=r["checkpoint"]
        w,d,data,predict,head=load(cfg,layout,reference);cache=cache_data(cfg,w,d,data);before=w.digest(True)
        for case in cases:
            controls=CONTROLS if part=="structure" else ("correct",)
            for seed in seeds:
                for control in controls:
                    s=list(state(cfg,cache,case,519,True,cfg["seed"]+case))
                    donor=cache["cache"][1-case][1] if case in (0,1) else cache["cache"][5-case][1]
                    s[3],s[4],s[5],original=endpoint(s[0],donor,control)
                    policy="quintic" if arm=="WarmQuintic" else "mean"
                    random_seed=89041+case if part=="p0" else cfg["seed"]+70000+case*100+seed
                    detailed=case in cfg["layer_trace_cases"] and seed==0 and control=="correct" and part!="gap"
                    arrays,receipt=sample(cfg,d,predict,s,random_seed,cache,detailed=detailed,policy=policy,score_known=original)
                    name=f"{case}_{arm}_{seed}_{control}";folder=out/name;folder.mkdir();path=folder/"trace.npz"
                    np.savez_compressed(path,**arrays);write(folder/"sampler.json",receipt)
                    if part=="p0" and arm=="WarmMean":
                        old=next(b for b in read(Path(cfg["parent"])/"rollout/result.json")["bindings"] if b["case"]==case and b["arm"]=="N2" and b["control"]=="correct")
                        z=np.load(checked(old["trace"]));assert np.array_equal(arrays["sample"],z["sample"]) and np.array_equal(arrays["raw_final"],z["raw_final"])
                    if replay:
                        old=Path(replay)/name;same_arrays(path,old/"trace.npz");assert receipt==read(old/"sampler.json")
                    bindings.append(dict(case=case,role="train" if case in cfg["train_cases"] else "dev_exposed",
                        arm=arm,layout=layout,seed=seed,control=control,task="inbetween",trace=ref(path),sampler=ref(folder/"sampler.json"),
                        source=next(r["left"] for r in cache["rows"] if r["case"]==case)))
                    print("R20_ROLLOUT",part,name,flush=True)
        assert w.digest(True)==before;predict.close()
    write(out/"result.json",dict(bindings=bindings,sampler_calls=len(bindings),optimizer_steps=0,
        part=part,whole_song_generations=0,operational_qualified=False))
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,part=part,dev_selection=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--part",choices=["p0","structure","gap","final"],required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.part,a.replay_from)

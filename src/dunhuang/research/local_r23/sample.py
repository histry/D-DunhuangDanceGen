"""32 new CFG1 samples plus four instrumented CFG2 exact reproductions."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.context_r21.sampler import sample
from dunhuang.research.branch_r22.engine import load,cache_data,state,model_meta
from .common import inputs,finish,arrays,read,fresh,forbid_optimizers
from .guidance import FixedGuidance

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();root,pc,pop=inputs(cfg)
    prior=read(root/"rollout/result.json");bindings=[];diagnostics=[];calls=0
    for arm in cfg["arms"]:
        model=read(root/f"models/{arm}/result.json")
        w,d,data,p,h=load(pc,root,arm,model["checkpoint"]);cache=cache_data(pc,root,d,data);before=model_meta(w,h)
        for case in pop["train_probe"]+pop["dev_cases"]:
            for variant in cfg["eval_variants"]:
                old=next(b for b in prior["bindings"] if b["arm"]==arm and b["case"]==case and b["variant"]==variant and b["control"]=="correct")
                oldtr=dict(np.load(checked(old["trace"])));oldreceipt=read(checked(old["sampler"]))
                for weight in cfg["guidance"]:
                    detailed=case in cfg["diagnostic_cases"] and variant==0
                    name=f"{arm}_{case}_{variant}_cfg{weight}"
                    if weight==2 and not detailed:
                        bindings.append(dict(arm=arm,case=case,variant=variant,cfg=weight,role=old["role"],source_group=old["source_group"],
                            trace=old["trace"],sampler=old["sampler"],reused_R22=True));continue
                    s,known,_=state(pc,cache,case,519,variant=variant)
                    predictor=FixedGuidance(p,weight,s,cache,pc["stage_snapshots"] if detailed else [])
                    tr,receipt=sample(pc,cache,predictor,s,old["seed"],keep_xt=detailed,score_mask=known)
                    receipt["cfg_weight"]=weight;calls+=1
                    assert receipt["noise_sha256"]==oldreceipt["noise_sha256"]
                    for key in ("condition","known","hard"):assert np.array_equal(tr[key],oldtr[key])
                    if weight==2:
                        for key in ("sample","raw_final"):assert np.array_equal(tr[key],oldtr[key]),(name,key)
                        for t in pc["stage_snapshots"]:
                            for label,old_label in (("unc","music_unconditional"),("cond","music_conditional")):
                                if str(t)+"_"+old_label in oldtr:assert np.array_equal(predictor.saved[str(t)+"_"+label],oldtr[str(t)+"_"+old_label])
                    arrays(out,name+".npz",dict(tr,**predictor.saved),replay)
                    if replay:assert receipt==read(Path(replay)/(name+"_sampler.json"))
                    write(out/(name+"_sampler.json"),receipt)
                    bindings.append(dict(arm=arm,case=case,variant=variant,cfg=weight,role=old["role"],source_group=old["source_group"],
                        trace_file=name+".npz",sampler_file=name+"_sampler.json",reused_R22=False))
                    if detailed:diagnostics.append(dict(arm=arm,case=case,variant=variant,cfg=weight,records=predictor.metrics,actual_sampling_states=True))
                    print("R23_SAMPLE",name,flush=True)
        assert model_meta(w,h)==before;p.close()
    assert calls==36 and len(bindings)==64
    finish(out,dict(bindings=bindings,diagnostics=diagnostics,sampler_calls=calls,new_cfg1_trajectories=32,
        detailed_cfg2_reproductions=4,cfg2_reused_without_sampling=28,optimizer_steps=0,operational_qualified=False),cfg,replay)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)

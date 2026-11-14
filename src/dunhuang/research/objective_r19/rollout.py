"""Single INBETWEEN, matched noise; no hidden future Root in FREE because FREE is not run."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.task_denoising_r18.rollout import sample
from .common import load,cache_data,state

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();arms=["D0"]+cfg["arms"]+(["U"] if read(root/"gate.json")["unique_admitted"] else [])
    bindings=[]
    for arm in arms:
        mode="unique" if arm=="U" else "self";reference=None if arm=="D0" else read(root/"models"/arm/"result.json")["checkpoint"]
        w,d,data,predict,head=load(cfg,mode,reference,adapt=arm!="D0")
        cache=cache_data(cfg,w,d,data);before=w.digest()
        for case in cfg["cases"]:
            y,m,o,v,k,h,meta,t,xt=state(cfg,cache,case,519,True,cfg["seed"]+case)
            for control in (("correct","typed_content") if arm=="U" else ("correct",)):
                arrays,receipt=sample(cfg,d,predict,m,v,k,h,meta,cfg["seed"]+70000+case,control)
                folder=out/f"{case}_{arm}_{control}";folder.mkdir();path=folder/"trace.npz"
                np.savez_compressed(path,**arrays);write(folder/"sampler.json",receipt)
                assert np.array_equal(arrays["sample"][arrays["hard"]],arrays["condition"][arrays["hard"]])
                if replay:
                    old=Path(replay)/folder.name;same_arrays(path,old/"trace.npz");assert receipt==read(old/"sampler.json")
                bindings.append(dict(case=case,arm=arm,control=control,task="inbetween",trace=ref(path),sampler=ref(folder/"sampler.json")))
                print("R19_ROLLOUT",folder.name,flush=True)
        assert w.digest()==before;predict.close()
    write(out/"result.json",dict(bindings=bindings,sampler_calls=len(bindings),optimizer_steps=0,
        whole_song_generations=0,operational_qualified=False))
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,task="inbetween",hard_real_edges=True,
        scope="Two Train sequences; not free dance or generated-history evaluation"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

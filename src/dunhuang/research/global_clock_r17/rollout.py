"""Task-separated225-frame A/B/C sampling; generated-only committed prefixes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from .engine import load,tensors,observed,global_sample
from .contract import metadata

def run(cfg,data_root,models,out,replay=None):
    out=fresh(out);forbid_optimizers();rows=[r for r in read(Path(data_root)/"result.json")["records"] if r["role"]=="eval"]
    models=Path(models);bindings=[]
    for arm,mode in (("A_numeric",None),("B_cross","cross"),("C_self","self")):
        reference=read(models/mode/"result.json")["checkpoint"] if mode else None
        wrapper,diffusion,data,predict,head=load(cfg,mode,reference);before=wrapper.digest()
        for r in rows:
            z,y,m=tensors(r);i=r["case"];meta=torch.tensor(metadata()[None],device="cuda")
            for task in cfg["tasks"]:
                prefix=None;prefix_ref=None
                if task=="progressive":
                    source=out/(str(i)+"_A_numeric_free")/"trace.npz";prefix_ref=ref(source)
                    prefix=np.load(checked(prefix_ref))["sample"][:75].copy()
                v,k,h=observed(y,task,prefix=prefix)
                arrays,receipt=global_sample(diffusion,predict,m,v,k,h,meta,cfg["seed"]+70000+i*100+cfg["tasks"].index(task))
                folder=out/(str(i)+"_"+arm+"_"+task);folder.mkdir();path=folder/"trace.npz"
                np.savez_compressed(path,**arrays);write(folder/"sampler.json",dict(receipt,prefix=prefix_ref,task=task,arm=arm))
                if task=="free":assert not arrays["known"][:,:13].any()
                if task=="progressive":
                    assert np.array_equal(arrays["sample"][:75,4:],prefix[:,4:]) and not arrays["known"][75:,:13].any()
                if replay:
                    old=Path(replay)/folder.name;same_arrays(path,old/"trace.npz")
                    # Prefix bytes and identity are exact; paths differ between replay roots.
                    oldr=read(old/"sampler.json");new=read(folder/"sampler.json")
                    oldprefix=oldr.pop("prefix");newprefix=new.pop("prefix");assert oldr==new
                    if oldprefix:assert oldprefix["sha256"]==newprefix["sha256"]
                bindings.append(dict(case=i,arm=arm,task=task,trace=ref(path),sampler=ref(folder/"sampler.json")))
                print("R17_ROLLOUT",arm,i,task,flush=True)
        assert wrapper.digest()==before;predict.close()
    write(out/"result.json",dict(bindings=bindings,sampler_calls=len(bindings),optimizer_steps=0,new_whole_song_generations=0,operational_qualified=False))
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,data=ref(Path(data_root)/"protocol.json"),
        models={mode:read(models/mode/"result.json")["checkpoint"] for mode in ("cross","self")},global_noise_single225=True,
        progressive_prefix="All arms use A_numeric FREE first75, never a future real target"))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--data",required=True);p.add_argument("--models",required=True)
    p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.data,a.models,a.out,a.replay_from)

"""Fresh original-sampler replay of every actually visited training cache."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from .trace import sample,trace_arrays
from .train import dataset

def run(cfg,root,out):
    root=Path(root);out=fresh(out);forbid_optimizers();z=dataset(root)
    d,n,s,f,e,info,a=runtime(cfg);music=torch.tensor(z["music"],device="cuda");target=torch.tensor(z["target"],device="cuda")
    reference=packet(list(z["raw139"][:,64:160,7:]));records=[];calls=0
    for arm in ("Q0","Q1"):
        model=read(root/f"models/{arm}/result.json")
        for cache in model["caches"]:
            weights=root/f"models/{arm}"/cache["weights"];assert sha(weights)==cache["weights_sha256"]
            restore(a,weights);assert digest(a)==cache["model_digest"]
            value,noise,trace=sample(d,a,reference,music,cache["noise"]["seeds"],trace=True,selected=cfg["cache_times"],target=target)
            arrays=trace_arrays(trace);indices=[arrays["times"].tolist().index(t) for t in cfg["cache_times"]]
            p=out/f"{arm}_{cache['step']}.npz";np.savez_compressed(p,xt=arrays["xt"][indices],times=np.array(cfg["cache_times"]))
            same_arrays(p,root/f"models/{arm}"/cache["file"]);assert noise==cache["noise"]
            records.append(dict(arm=arm,step=cache["step"],state_kind="ACTUAL_REVERSE_XT",used_for_loss=cache["used_for_loss"],exact=True,noise=noise));calls+=1
    write(out/"result.json",dict(records=records,sampler_calls=calls,generated_trajectories=4*calls,optimizer_steps=0))
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,files={p.name:sha(p) for p in out.glob("*.npz")}))
    a.close();print("R27_VISITED_CACHES_REPLAYED",calls,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(read(a.config),a.root,a.out)

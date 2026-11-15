"""New-noise model-internal source recovery; labels are generated targets, not human truth."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from .trace import sample
from .train import dataset
from .qualification import sampling_gate

def run(cfg,root,out,models,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();z=dataset(root)
    d,n,s,f,e,info,adapter=runtime(cfg);frozen=digest(d.model)
    music=torch.tensor(z["music"],device="cuda");target=torch.tensor(z["target"],device="cuda")
    refs={mode:packet(list(z["raw139"][[1,0,3,2] if mode=="wrong" else [0,1,2,3],64:160,7:])) for mode in ("correct","wrong")}
    arrays={};records=[];noise_rows=[];calls=0;baseline={}
    def measure(name,mode,k,value):
        with torch.no_grad():points=f(n.unnormalize(value.clone()),s)[:,:,:22]
        source_errors=decoded_metrics(value,target,n,s,f)
        for i,error in enumerate(source_errors):
            quality=motion_metrics(n.unnormalize(value.clone())[i].cpu().numpy(),points[i].cpu().numpy(),np.zeros(256,bool))
            records.append(dict(arm=name,mode=mode,teacher=i,seed_index=k,metrics=error,quality=quality,
                target_scope="MODEL_GENERATED_FULL139",C4_truth=False))
        arrays[f"{name}_{mode}_{k}_normalized"]=value.cpu().numpy()
        arrays[f"{name}_{mode}_{k}_points"]=points.cpu().numpy()
    for k,seeds in enumerate(cfg["eval_noise_seeds"]):
        value,noise,_=sample(d,adapter,None,music,seeds);calls+=1;baseline[k]=value.clone();measure("baseline","none",k,value)
        noise_rows.append(dict(arm="baseline",mode="none",seed_index=k,noise=noise))
    for arm in models:
        adapter.close()
        capacity=arm=="P2"
        adapter=ReferenceModel(capacity).cuda()
        if arm=="R26_C1":
            path=Path(cfg["r26"])/"models/C1/adapter.pt"
            assert sha(path)==read("reports/r26_release/r26_summary.json")["models"]["C1"]["checkpoint_hashes"]["adapter.pt"]
            adapter.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["state"],strict=True)
        else:restore(adapter,root/f"models/{arm}/adapter.pt")
        adapter.attach(d.model)
        for mode in (("correct","wrong") if arm=="R26_C1" else ("correct","wrong","none")):
            for k,seeds in enumerate(cfg["eval_noise_seeds"]):
                value,noise,_=sample(d,adapter,None if mode=="none" else refs[mode],music,seeds);calls+=1
                if mode=="none":assert torch.equal(value,baseline[k])
                measure(arm,mode,k,value);noise_rows.append(dict(arm=arm,mode=mode,seed_index=k,noise=noise))
                print("R27_POSITIVE_SAMPLE",arm,mode,k,flush=True)
        assert digest(d.model)==frozen
    result=dict(records=records,models=models,gates={a:sampling_gate(records,a) for a in models},noise=noise_rows,
        sampler_calls=calls,generated_trajectories=4*calls,teacher_eval_seed_disjoint=True,reference_state_write=False,no_reference_exact=True,
        optimizer_steps=0,full_native_status="NOT_EVALUABLE",operational_qualified=False)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    adapter.close();print("R27_POSITIVE_GATES",{a:r["passed"] for a,r in result["gates"].items()},flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--models",nargs="+",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.models,a.replay_from)

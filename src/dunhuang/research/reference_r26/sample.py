"""Same official sampler and noise as R24; neural control only, no pose state writes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.research.lodge_adapter_r25.sample import sample
from dunhuang.research.lodge_adapter_r25.replay import change_conditions
from .common import *

def run(cfg,root,out,replay=None):
    out=fresh(out);root=Path(root);forbid_optimizers();pop,data=dataset(root)
    oldroot=Path(cfg["r24"]);seal=read("reports/r24_release/r24_receipt.json")
    for name,h in seal["evidence"].items():assert sha(oldroot/name)==h
    z=dict(np.load(checked(read(oldroot/"main/protocol.json")["arrays"])))
    d,n,s,f,e,info,adapter=runtime(cfg,"C0");frozen=digest(d.model);arrays={};receipts=[];calls=0
    conditions=change_conditions()
    def switch(arm):
        nonlocal adapter
        adapter.close();torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
        adapter=Adapter(arm!="C0",width=64,part_width=16,layers=(1,4,7),hidden=d.model.latent_dim).cuda()
        adapter.attach(d.model)
        restore(adapter,root/f"models/{arm}/adapter.pt",cfg,arm)
    for arm in cfg["arms"]:
        switch(arm)
        assert sha(root/f"models/{arm}/adapter.pt")==read(root/f"models/{arm}/result.json")["checkpoint_hashes"]["adapter.pt"]
        modes=["correct","swapped"]+(["occluded","music_swapped"] if arm=="C2" else [])
        for mode in modes:
            ref=packet([z["source139"][1-max(ev,0) if mode=="swapped" else max(ev,0),64:160,7:] for m,ev in conditions],
                [cfg["seed"]+i for i in range(6)],[mode=="occluded"]*6,[ev<0 for m,ev in conditions])
            music=torch.tensor(np.stack([z["music"][1-m if mode=="music_swapped" else m] for m,ev in conditions]),device="cuda")
            for seed in cfg["eval_seeds"]:
                value,noise=sample(d,adapter,ref,music,seed,arm);calls+=1
                for i in (0,3):
                    j=3-i if mode=="music_swapped" else i
                    assert np.array_equal(value[i].cpu().numpy(),z[f"{seed}_normalized"][j])
                raw=n.unnormalize(value.clone());points=f(raw,s)[:,:,:22]
                stem=f"change_{arm}_{mode}_{seed}"
                for name,arr in [("normalized",value),("139",raw),("22",points)]:arrays[stem+"_"+name]=arr.cpu().numpy()
                receipts.append(dict(dataset="change",arm=arm,mode=mode,seed=seed,stem=stem,conditions=conditions,noise=noise))
                print("R26_CHANGE_SAMPLE",arm,mode,seed,flush=True)
        assert digest(d.model)==frozen
    cases=pop["train_probe"]+pop["dev_probe"]
    for arm,mode in [("baseline","none")]+[(a,m) for a in cfg["arms"] for m in ("correct","swapped")]+[("C2",m) for m in ("removed","occluded","music_swapped")]:
        switch("C2" if arm=="baseline" else arm)
        donor=cases[1:]+cases[:1] if mode=="swapped" else cases
        ref=None if mode in ("none","removed") else packet([data[c]["raw139"][64:160,7:] for c in donor],[cfg["seed"]+c for c in cases],[mode=="occluded"]*len(cases))
        music=torch.tensor(np.stack([data[c]["music"] for c in cases]),device="cuda")
        if mode=="music_swapped":music=music.roll(1,0)
        value,noise=sample(d,adapter,ref,music,26091,arm);calls+=1
        raw=n.unnormalize(value.clone());points=f(raw,s)[:,:,:22];stem=f"paired_{arm}_{mode}"
        for name,arr in [("normalized",value),("139",raw),("22",points)]:arrays[stem+"_"+name]=arr.cpu().numpy()
        if mode=="removed":assert np.array_equal(arrays[stem+"_normalized"],arrays["paired_baseline_none_normalized"])
        receipts.append(dict(dataset="paired",arm=arm,mode=mode,stem=stem,cases=cases,seed=26091,noise=noise))
        assert digest(d.model)==frozen
        print("R26_PAIRED_SAMPLE",arm,mode,flush=True)
    assert calls==26
    path=out/"arrays.npz";np.savez_compressed(path,**arrays)
    result=dict(receipts=receipts,sampler_calls=calls,generated_trajectories=166,null_reference_exact=True,
        C3_trained=False,reference_state_write=False,optimizer_steps=0,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/"arrays.npz")
    adapter.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

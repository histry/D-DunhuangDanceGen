"""Frozen and trained comparisons, with pure neural reference control and no state writes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write,sha
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.lodge_reference_r24.noise import common_noise
from .common import runtime,dataset,packet,install,restore
from .model import digest

def sample(d,adapter,ref,music,seed,arm):
    zeros=torch.zeros(len(music),256,139,device="cuda")
    adapter.clear()
    if ref is not None:install(adapter,ref)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    with common_noise(seed,batch=len(music),device="cuda") as receipt:
        value=d.inpaint_soft_ddim((len(music),256,139),music,None,constraint=dict(value=zeros,mask=zeros))
    adapter.clear();assert receipt["draws"]==50
    return value,dict(receipt,reference_state_write=False,reference_shared_across_music_CFG=True,arm=arm)

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    seal=read("reports/r24_release/r24_receipt.json");parent=Path(cfg["parent"])
    for name,h in seal["evidence"].items():assert sha(parent/name)==h
    r24=read(parent/"main/result.json");z=dict(np.load(checked(read(parent/"main/protocol.json")["arrays"])))
    pop,data=dataset(root);d,n,s,fk,e,info,adapter=runtime(cfg);frozen=digest(d.model);adapter.attach(d.model)
    arrays={};receipts=[];calls=0
    conditions=[(m,event) for m in range(2) for event in (-1,0,1)]
    music=torch.tensor(np.stack([z["music"][m] for m,event in conditions]),device="cuda")
    for arm in cfg["arms"]:
        assert sha(root/f"models/{arm}/adapter.pt")==read(root/f"models/{arm}/result.json")["checkpoint_hashes"]["adapter.pt"]
        restore(adapter,root/f"models/{arm}/adapter.pt",cfg,arm)
        modes=["correct"] if arm=="event" else ["correct","phase_removed","phase_wrong","reference_swapped"]
        for mode in modes:
            rotations=[z["source139"][max(event,0),64:160,7:] for m,event in conditions]
            if mode=="reference_swapped":rotations=[z["source139"][1-max(event,0),64:160,7:] for m,event in conditions]
            control="removed" if mode=="phase_removed" else "wrong" if mode=="phase_wrong" else "correct"
            ref=packet(rotations,drops=[event<0 for m,event in conditions],control=control)
            for seed in cfg["eval_seeds"]:
                value,receipt=sample(d,adapter,ref,music,seed,arm);calls+=1
                assert receipt["sha256"]==next(x["sha256"] for x in r24["noise_receipts"] if x["seed"]==seed)
                # Empty reference must exactly preserve the same pretrained sampler.
                for i in (0,3):assert np.array_equal(value[i].cpu().numpy(),z[str(seed)+"_normalized"][i]),("Null-reference baseline",arm,mode,seed)
                raw=n.unnormalize(value.clone());points=fk(raw,s)[:,:,:22]
                stem=f"change_{arm}_{mode}_{seed}";arrays[stem+"_normalized"]=value.cpu().numpy();arrays[stem+"_139"]=raw.cpu().numpy();arrays[stem+"_22"]=points.cpu().numpy()
                receipts.append(dict(dataset="change",arm=arm,mode=mode,seed=seed,stem=stem,conditions=conditions,noise=receipt))
                print("R25_CHANGE_SAMPLE",arm,mode,seed,flush=True)
    cases=cfg["paired_eval_cases"];music=torch.tensor(np.stack([data[c]["music"] for c in cases]),device="cuda")
    for arm,mode in [("baseline","none"),("event","correct"),("phase","correct"),("phase","removed"),("phase","reference_swapped"),("phase","phase_removed"),("phase","phase_wrong"),("phase","occluded")]:
        if arm!="baseline":restore(adapter,root/f"models/{arm}/adapter.pt",cfg,arm)
        ref=None
        if mode not in ("none","removed"):
            donor=cases[1:]+cases[:1] if mode=="reference_swapped" else cases
            ref=packet([data[c]["raw139"][64:160,7:] for c in donor],seeds=[cfg["seed"]+c for c in cases],augment=mode=="occluded",
                control="removed" if mode=="phase_removed" else "wrong" if mode=="phase_wrong" else "correct")
        value,receipt=sample(d,adapter,ref,music,cfg["paired_eval_seed"],arm);calls+=1
        raw=n.unnormalize(value.clone());points=fk(raw,s)[:,:,:22];stem=f"paired_{arm}_{mode}"
        arrays[stem+"_normalized"]=value.cpu().numpy();arrays[stem+"_139"]=raw.cpu().numpy();arrays[stem+"_22"]=points.cpu().numpy()
        if mode=="removed":assert np.array_equal(arrays[stem+"_normalized"],arrays["paired_baseline_none_normalized"])
        receipts.append(dict(dataset="paired",arm=arm,mode=mode,seed=cfg["paired_eval_seed"],stem=stem,cases=cases,noise=receipt))
        print("R25_PAIRED_SAMPLE",arm,mode,flush=True)
    assert digest(d.model)==frozen and calls==18
    path=out/"arrays.npz";np.savez_compressed(path,**arrays)
    result=dict(receipts=receipts,sampler_calls=calls,change_trajectories=60,paired_trajectories=64,total_trajectories=124,
        null_reference_exact=True,frozen_model_unchanged=True,optimizer_steps=0,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/"arrays.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),data=str(root/"data/result.json"),arrays_sha256=sha(path),checkpoint=info))
    adapter.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

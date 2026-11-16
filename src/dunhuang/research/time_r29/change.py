"""Exposed unpaired ChangE clock controls with neural references only."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import runtime
from .model import mapping
from .dynamics import decomposition
from dunhuang.research.contract_r28.representation import packet
from dunhuang.research.state_r27.common import read,write,sha,checked,fresh,commit,forbid_optimizers,restore,digest,metrics,motion_metrics,same_arrays
from dunhuang.research.lodge_adapter_r25.sample import sample
from dunhuang.research.lodge_adapter_r25.replay import change_conditions
from dunhuang.research.contract_r28.input_contract import foot_proxy

def run(cfg,root,out,replay=None,arms=("B1","F1")):
    root=Path(root);out=fresh(out);forbid_optimizers()
    old=dict(np.load(checked(read(Path(cfg["r24"])/"main/protocol.json")["arrays"])))
    baselinepath=Path(cfg["r28"])/"change/arrays.npz";assert sha(baselinepath)==read(Path(cfg["r28"])/"artifacts_manifest.json")["files"]["change/arrays.npz"]["sha256"]
    hist=np.load(baselinepath);conditions=change_conditions();arrays={};records=[];noises=[];calls=0
    d,n,s,f,e,info,a=runtime(cfg);frozen=digest(d.model)
    music=torch.tensor(np.stack([old["music"][m] for m,event in conditions]),device="cuda")
    def measure(arm,mode,raw,points,normalized,reused=False):
        stem=arm+"_"+mode
        for name,value in (("139",raw),("22",points),("normalized",normalized)):arrays[stem+"_"+name]=value
        for i,(m,ev) in enumerate(conditions):
            q=motion_metrics(raw[i],points[i],np.zeros(256,bool));q.pop("foot_geometry_proxy",None)
            records.append(dict(arm=arm,mode=mode,music=m,event=ev,reused=reused,errors=[metrics(raw[i],points[i],r,p) for r,p in zip(old["source139"],old["source22"])],
                quality=q,floor=foot_proxy(points[i]),dynamics=decomposition(points[i]),truth_scope="RETIMED_SOURCE_PROXY_NOT_NEW_MUSIC_PAIRED_TRUTH"))
    for mode,oldmode in (("correct","correct"),("swapped","swapped")):measure("B0",mode,hist["T1_"+oldmode+"_139"],hist["T1_"+oldmode+"_22"],hist["T1_"+oldmode+"_normalized"],True)
    for arm in arms:
        restore(a,root/f"models/{arm}/adapter.pt")
        for mode in ("correct","wrong_clock","swapped"):
            ref=packet([old["source139"][1-max(ev,0) if mode=="swapped" else max(ev,0),64:160,7:] for m,ev in conditions],drops=[ev<0 for m,ev in conditions])
            ref["mapped_source"]=mapping(6,"cuda",wrong=mode=="wrong_clock")
            pred,noise=sample(d,a,ref,music,24041,arm);calls+=1
            for i in (0,3):assert np.array_equal(pred[i].cpu().numpy(),old["24041_normalized"][i])
            raw=n.unnormalize(pred.clone());p=f(raw,s)[:,:,:22]
            measure(arm,mode,raw.cpu().numpy(),p.cpu().numpy(),pred.cpu().numpy());noises.append(dict(arm=arm,mode=mode,noise=noise))
            print("R29_CHANGE",arm,mode,flush=True)
    assert digest(d.model)==frozen
    result=dict(records=records,noise=noises,sampler_calls=calls,generated_trajectories=6*calls,reused_trajectories=12,
        NoRef_original_exact=True,reference_state_write=False,optimizer_steps=0,Source_Root_unknown=True,full_native_status="NOT_EVALUABLE",operational_qualified=False,whole_song_generations=0)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    a.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");p.add_argument("--arms",nargs="+",choices=["B1","F1","COVER"],default=["B1","F1"]);a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from,a.arms)

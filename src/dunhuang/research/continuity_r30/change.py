"""Fixed two-event/two-music transfer; world Root and contact truth remain unknown."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import runtime,checked_parent
from .metrics import geometry,absolute,paired
from dunhuang.research.contract_r28.representation import packet
from dunhuang.research.time_r29.model import mapping
from dunhuang.research.time_r29.dynamics import decomposition
from dunhuang.research.lodge_adapter_r25.sample import sample
from dunhuang.research.lodge_adapter_r25.replay import change_conditions
from dunhuang.research.state_r27.common import read,write,sha,checked,fresh,commit,forbid_optimizers,digest,restore,motion_metrics,same_arrays
from dunhuang.research.contract_r28.input_contract import foot_proxy

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    old=dict(np.load(checked(read("outputs/lodge_reference_r24_01/main/protocol.json")["arrays"])))
    r0=np.load(checked_parent(cfg,"change_coverage/arrays.npz"))
    d,n,s,f,e,info,a=runtime(cfg);frozen=digest(d.model);conditions=change_conditions()
    music=torch.tensor(np.stack([old["music"][m] for m,event in conditions]),device="cuda")
    arrays={};records=[];noises=[];calls=0
    def add(arm,mode,raw,points,pred,reused=False):
        for key,value in (("139",raw),("22",points),("normalized",pred)):arrays[arm+"_"+mode+"_"+key]=value
        for i,(m,event) in enumerate(conditions):
            q=motion_metrics(raw[i],points[i],np.zeros(256,bool));q.pop("foot_geometry_proxy",None)
            records.append(dict(arm=arm,mode=mode,music=m,event=event,reused=reused,
                errors=[geometry(raw[i],points[i],r,p) for r,p in zip(old["source139"],old["source22"])],
                source_absolute=[absolute(p) for p in old["source22"]],generated_absolute=absolute(points[i]),
                paired_root_relative_proxy=[paired(points[i],p)["Root_relative_body"] for p in old["source22"]],
                quality=q,floor=foot_proxy(points[i]),dynamics=decomposition(points[i]),
                Root_truth="UNKNOWN",contact_truth="UNKNOWN",scope="RETIMED_SOURCE_PROXY; NOT_NEW_MUSIC_PAIRED_TRUTH"))
    for mode in ("correct","wrong_clock","swapped"):
        stem="COVER_"+mode;add("R0",mode,r0[stem+"_139"],r0[stem+"_22"],r0[stem+"_normalized"],True)
    for arm in ("R1","R2"):
        if not (root/f"models/{arm}/result.json").exists():continue
        model=read(root/f"models/{arm}/result.json");a.aligned=model["aligned"];restore(a,root/f"models/{arm}/adapter.pt")
        for mode in ("correct","wrong_clock","swapped"):
            ref=packet([old["source139"][1-max(event,0) if mode=="swapped" else max(event,0),64:160,7:] for m,event in conditions],drops=[event<0 for m,event in conditions])
            ref["mapped_source"]=mapping(6,"cuda",wrong=mode=="wrong_clock")
            pred,nr=sample(d,a,ref,music,24041,arm);calls+=1
            for i in (0,3):assert np.array_equal(pred[i].cpu().numpy(),old["24041_normalized"][i])
            raw=n.unnormalize(pred.clone())
            with torch.no_grad():points=f(raw,s)[:,:,:22]
            add(arm,mode,raw.cpu().numpy(),points.cpu().numpy(),pred.cpu().numpy());noises.append(dict(arm=arm,mode=mode,noise=nr))
            print("R30_CHANGE",arm,mode,flush=True)
    assert digest(d.model)==frozen;a.close();np.savez_compressed(out/"arrays.npz",**arrays)
    result=dict(records=records,noise=noises,sampler_calls=calls,generated_trajectories=6*calls,reused_trajectories=18,optimizer_steps=0,
        NoRef_original_exact=True,reference_state_write=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE",operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,arrays_sha256=sha(out/"arrays.npz")))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

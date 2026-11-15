"""Fixed historical ChangE transfer after real-source training; never used for selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .representation import packet
from .input_contract import foot_proxy
from dunhuang.research.state_r27.common import read,write,sha,checked,fresh,forbid_optimizers,runtime,restore,digest,metrics,motion_metrics,commit,same_arrays
from dunhuang.research.lodge_adapter_r25.sample import sample
from dunhuang.research.lodge_adapter_r25.replay import change_conditions

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    old=dict(np.load(checked(read(Path(cfg["r24"])/"main/protocol.json")["arrays"])))
    histpath=Path(cfg["r26"])/"samples/arrays.npz";manifest=read(Path(cfg["r26"])/"artifacts_manifest.json")["files"]
    assert sha(histpath)==manifest["samples/arrays.npz"]["sha256"]
    hist=dict(np.load(histpath));conditions=change_conditions()
    d,n,s,f,e,info,a=runtime(read("src/dunhuang/research/state_r27/config.json"));frozen=digest(d.model)
    music=torch.tensor(np.stack([old["music"][m] for m,ev in conditions]),device="cuda");arrays={};records=[];noises=[];calls=0
    def measure(arm,mode,raw,points,normalized):
        stem=arm+"_"+mode;arrays[stem+"_139"]=raw;arrays[stem+"_22"]=points;arrays[stem+"_normalized"]=normalized
        for i,(m,ev) in enumerate(conditions):
            q=motion_metrics(raw[i],points[i],np.zeros(256,bool));q.pop("foot_geometry_proxy",None)
            records.append(dict(arm=arm,mode=mode,music=m,event=ev,
                errors=[metrics(raw[i],points[i],x,p) for x,p in zip(old["source139"],old["source22"])],
                quality=q,floor=foot_proxy(points[i]),source_truth="RETIMED_REFERENCE_PROXY_NOT_TRUE_NEW_MUSIC_PAIR"))
    for mode in ("correct","swapped"):
        stem=f"change_C1_{mode}_24041";measure("R26_C1",mode,hist[stem+"_139"],hist[stem+"_22"],hist[stem+"_normalized"])
    for arm in ("P","T0","T1"):
        restore(a,(Path(cfg["r27"])/"models/P/adapter.pt") if arm=="P" else root/f"models/{arm}/adapter.pt")
        for mode in ("correct","swapped"):
            ref=packet([old["source139"][1-max(ev,0) if mode=="swapped" else max(ev,0),64:160,7:] for m,ev in conditions],drops=[ev<0 for m,ev in conditions])
            v,noise=sample(d,a,ref,music,24041,arm);calls+=1
            for i in (0,3):assert np.array_equal(v[i].cpu().numpy(),old["24041_normalized"][i])
            raw=n.unnormalize(v.clone());p=f(raw,s)[:,:,:22]
            measure(arm,mode,raw.cpu().numpy(),p.cpu().numpy(),v.cpu().numpy());noises.append(dict(arm=arm,mode=mode,noise=noise))
            print("R28_CHANGE_SAMPLE",arm,mode,flush=True)
        assert digest(d.model)==frozen
    comparisons=[]
    for arm in ("R26_C1","P","T0","T1"):
        for m in range(2):
            events=[]
            for ev in range(2):
                def get(mode,event):return next(r for r in records if (r["arm"],r["mode"],r["music"],r["event"])==(arm,mode,m,event))
                c=get("correct",ev);w=get("swapped",ev);b=get("correct",-1);x,y,z=c["errors"][ev],w["errors"][ev],b["errors"][ev]
                values=dict(body_gain=x["body22_rad"]<=.75*z["body22_rad"],body_reference=x["body22_rad"]<y["body22_rad"],
                    local_reference=x["pelvis_wrist_rmse_m"]<y["pelvis_wrist_rmse_m"],
                    activity=x["body_activity_ratio"] is not None and .6<=x["body_activity_ratio"]<=1.6,
                    extent=x["arm_range_ratio"] is not None and .6<=x["arm_range_ratio"]<=1.6)
                q=c["quality"];base=b["quality"]
                quality=dict(global_jerk=q["jerk_p95_mps3"]<=1.1*base["jerk_p95_mps3"],core_jerk=q["regions"]["core_interior_unprompted"]["p95"]<=1.1*base["regions"]["core_interior_unprompted"]["p95"])
                events.append(dict(event=ev,tests=values,source_pass=all(values.values()),jerk_preserved=all(quality.values()),quality=quality))
            comparisons.append(dict(arm=arm,music=m,events=events,source_pass=all(x["source_pass"] for x in events),source_and_jerk_pass=all(x["source_pass"] and x["jerk_preserved"] for x in events)))
    result=dict(records=records,comparisons=comparisons,noise=noises,sampler_calls=calls,generated_trajectories=6*calls,reused_trajectories=12,
        no_reference_exact=True,optimizer_steps=0,selection_on_ChangE=False,
        source_gate_scope="R28_BODY_LOCAL_GEOMETRY_AMPLITUDE_GATE; NOT_REPLACEMENT_FOR_R26_STRICT_GATE",
        full_native_status="NOT_EVALUABLE",whole_song_generations=0,operational_qualified=False)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    a.close();print("R28_CHANGE_GATES",[(r["arm"],r["music"],r["source_and_jerk_pass"]) for r in comparisons],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.root,args.out,args.replay_from)

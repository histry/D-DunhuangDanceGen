"""Fixed full sampling on trained and adaptation-held-out real sources, no DEV selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .representation import packet
from .input_contract import foot_proxy
from .train import dataset
from dunhuang.research.state_r27.common import read,sha,write,fresh,forbid_optimizers,runtime,digest,restore,metrics,motion_metrics,commit,same_arrays
from dunhuang.research.state_r27.trace import sample

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    d,n,s,f,e,info,a=runtime(read("src/dunhuang/research/state_r27/config.json"));frozen=digest(d.model)
    arrays={};records=[];noises=[];baseline={};calls=0
    def measure(role,arm,mode,k,v,rows):
        raw=n.unnormalize(v.clone());p=f(raw,s)[:,:,:22];stem=f"{role}_{arm}_{mode}_{k}"
        arrays[stem+"_normalized"]=v.cpu().numpy();arrays[stem+"_139"]=raw.cpu().numpy();arrays[stem+"_22"]=p.cpu().numpy()
        for i,r in enumerate(rows):
            quality=motion_metrics(raw[i].cpu().numpy(),p[i].cpu().numpy(),np.zeros(256,bool));quality.pop("foot_geometry_proxy",None)
            records.append(dict(role=role,arm=arm,mode=mode,seed_index=k,item=i,case=(cfg["train_cases"] if role=="train" else cfg["holdout_cases"])[i],
                metrics=metrics(raw[i].cpu().numpy(),p[i].cpu().numpy(),r["model_raw"],r["model_points"]),quality=quality,
                floor=foot_proxy(p[i].cpu().numpy()),root_rmse_m=float(np.sqrt(np.mean((raw[i,64:160,4:7].cpu().numpy()-r["model_raw"][64:160,4:7])**2))),
                target_scope="ORIGINAL_REAL_SOURCE_IN_FIXED_PUBLISHED_MODEL0_COORDINATES; PREDICTION_OFFICIAL_CLIPPED",
                generated_motion_height_refit=False))
    for role,cases in (("train",cfg["train_cases"]),("holdout_exposed_R26",cfg["holdout_cases"])):
        rows=dataset(root,cases);music=torch.tensor(np.stack([x["music"] for x in rows]),device="cuda")
        for k,seeds in enumerate(cfg["real_eval_seeds"]):
            v,noise,_=sample(d,a,None,music,seeds);calls+=1;baseline[role,k]=v.clone();measure(role,"baseline","none",k,v,rows)
            noises.append(dict(role=role,arm="baseline",mode="none",seed_index=k,noise=noise))
    for arm in ("R26_C1","T0","T1"):
        if arm=="R26_C1":
            state=torch.load(Path(cfg["r26"])/"models/C1/adapter.pt",map_location="cpu",weights_only=True)["state"];a.load_state_dict(state,strict=True)
        else:restore(a,root/f"models/{arm}/adapter.pt")
        for role,cases in (("train",cfg["train_cases"]),("holdout_exposed_R26",cfg["holdout_cases"])):
            rows=dataset(root,cases);music=torch.tensor(np.stack([x["music"] for x in rows]),device="cuda");refs=[x["model_raw"][64:160,7:] for x in rows]
            for mode in (("correct","wrong") if arm=="R26_C1" else ("correct","wrong","none")):
                ref=None if mode=="none" else packet(refs if mode=="correct" else refs[1:]+refs[:1])
                for k,seeds in enumerate(cfg["real_eval_seeds"]):
                    v,noise,_=sample(d,a,ref,music,seeds);calls+=1
                    if mode=="none":assert torch.equal(v,baseline[role,k])
                    measure(role,arm,mode,k,v,rows);noises.append(dict(role=role,arm=arm,mode=mode,seed_index=k,noise=noise))
                print("R28_REAL_SAMPLE",arm,role,mode,flush=True)
        assert digest(d.model)==frozen
    gates=[]
    for role in ("train","holdout_exposed_R26"):
        for arm in ("R26_C1","T0","T1"):
            conditions=[]
            for i in range(4):
                for k in range(2):
                    def get(a,m):return next(x for x in records if (x["role"],x["arm"],x["mode"],x["item"],x["seed_index"])==(role,a,m,i,k))
                    correct=get(arm,"correct");wrong=get(arm,"wrong");none=get("baseline","none");x,y,z=correct["metrics"],wrong["metrics"],none["metrics"]
                    vals=dict(body_gain=x["body22_rad"]<=.75*z["body22_rad"],body_reference=x["body22_rad"]<y["body22_rad"],
                        local_reference=x["pelvis_wrist_rmse_m"]<y["pelvis_wrist_rmse_m"],
                        activity=x["body_activity_ratio"] is not None and .6<=x["body_activity_ratio"]<=1.6,
                        extent=x["arm_range_ratio"] is not None and .6<=x["arm_range_ratio"]<=1.6)
                    q,b=correct["quality"],none["quality"]
                    proxy=dict(global_jerk=q["jerk_p95_mps3"]<=1.1*b["jerk_p95_mps3"],
                        core_jerk=q["regions"]["core_interior_unprompted"]["p95"]<=1.1*b["regions"]["core_interior_unprompted"]["p95"],
                        root_speed=q["root_speed_p95_mps"]<=1.1*b["root_speed_p95_mps"],
                        minimum_floor_height=correct["floor"]["minimum_height_m"]>=none["floor"]["minimum_height_m"]-.02)
                    conditions.append(dict(item=i,seed_index=k,tests=vals,source_pass=all(vals.values()),proxy_preservation=proxy,proxy_pass=all(proxy.values())))
            source_count=sum(all(c["source_pass"] for c in conditions if c["item"]==i) for i in range(4))
            proxy_count=sum(all(c["proxy_pass"] for c in conditions if c["item"]==i) for i in range(4))
            gates.append(dict(role=role,arm=arm,conditions=conditions,source_recordings_passed=source_count,source_pass=source_count>=3,
                proxy_recordings_preserved=proxy_count,proxy_pass=proxy_count>=3,full_native_status="NOT_EVALUABLE"))
    result=dict(records=records,gates=gates,noise=noises,sampler_calls=calls,generated_trajectories=4*calls,no_reference_exact=True,
        optimizer_steps=0,historical_source_exposure_disclosed=True,all_arm_outputs_use_common_model0_readout=True,
        foot_height_proxy_not_contact_truth=True,full_native_status="NOT_EVALUABLE",whole_song_generations=0,operational_qualified=False)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    a.close();print("R28_REAL_GATES",[(x["arm"],x["role"],x["source_recordings_passed"],x["proxy_recordings_preserved"]) for x in gates],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.root,args.out,args.replay_from)

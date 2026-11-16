"""Actual clock/reference interventions; original B0 is reused, never retrained."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import runtime,rows,packet
from .dynamics import decomposition,rotation_groups
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,restore,digest,metrics,motion_metrics,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.trace import sample,trace_arrays
from dunhuang.research.contract_r28.input_contract import foot_proxy
from dunhuang.research.reference_r26.evaluate import timing

def run(cfg,root,out,domain,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    d,n,s,f,e,info,a=runtime(cfg,0);frozen=digest(d.model);
    arrays={};records=[];noises=[];calls=0;reused=0;baseline={}
    def measure(role,arm,mode,k,raw,points,normalized,values,items,reuse=False):
        stem=f"{role}_{arm}_{mode}_{k}"
        arrays[stem+"_normalized"]=normalized;arrays[stem+"_139"]=raw;arrays[stem+"_22"]=points
        for i,row in enumerate(values):
            quality=motion_metrics(raw[i],points[i],np.zeros(256,bool));quality.pop("foot_geometry_proxy",None)
            records.append(dict(domain=domain,role=role,arm=arm,mode=mode,seed_index=k,item=i,source_item=items[i],
                recording=str(items[i]).split(":")[0] if domain=="fine" else str(items[i]),reused=reuse,
                metrics=metrics(raw[i],points[i],row["model_raw"],row["model_points"]),quality=quality,floor=foot_proxy(points[i]),
                dynamics=decomposition(points[i]),rotation_groups=rotation_groups(raw[i]),timing_proxy=timing(raw[i],row["model_raw"]),
                target_scope="ORIGINAL_SOURCE_COMMON_MODEL0; CORRECT_IDENTITY_CLOCK_TRUTH_NOT_EVENT_SEMANTICS"))
    if domain=="aist":
        parent=Path(cfg["r28"]);path=parent/"evaluation/arrays.npz"
        assert sha(path)==read(parent/"artifacts_manifest.json")["files"]["evaluation/arrays.npz"]["sha256"]
        old=np.load(path)
        for role,oldrole in (("train","train"),("holdout","holdout_exposed_R26")):
            values,items=rows(cfg,root,domain,role)
            for k in range(2):
                for arm,mode,stem in [("baseline","none",f"{oldrole}_baseline_none_{k}")]+[("B0",m,f"{oldrole}_T1_{o}_{k}") for m,o in (("correct","correct"),("swapped","wrong"))]:
                    measure(role,arm,mode,k,old[stem+"_139"],old[stem+"_22"],old[stem+"_normalized"],values,items,True);reused+=4
                    if arm=="baseline":baseline[role,k]=old[stem+"_normalized"]
    else:
        for role in ("train","holdout"):
            values,items=rows(cfg,root,domain,role);music=torch.tensor(np.stack([x["music"] for x in values]),device="cuda")
            for mode in ("none","correct","swapped"):
                for k,seeds in enumerate(cfg["real_eval_seeds"]):
                    pred,noise,_=sample(d,a,None if mode=="none" else packet(values,mode),music,seeds);calls+=1
                    raw=n.unnormalize(pred.clone());p=f(raw,s)[:,:,:22]
                    measure(role,"baseline" if mode=="none" else "F0",mode,k,raw.cpu().numpy(),p.cpu().numpy(),pred.cpu().numpy(),values,items)
                    noises.append(dict(role=role,arm="F0",mode=mode,seed_index=k,noise=noise))
                    if mode=="none":baseline[role,k]=pred.cpu().numpy()
    arm="B1" if domain=="aist" else "F1";a.beta=cfg["beta"];restore(a,root/f"models/{arm}/adapter.pt")
    for role in ("train","holdout"):
        values,items=rows(cfg,root,domain,role);music=torch.tensor(np.stack([x["music"] for x in values]),device="cuda")
        for mode in ("correct","wrong_clock","swapped","none"):
            for k,seeds in enumerate(cfg["real_eval_seeds"]):
                watch=domain=="aist" and role=="train" and mode=="correct" and k==0
                pred,noise,trace=sample(d,a,None if mode=="none" else packet(values,mode),music,seeds,trace=watch,selected=cfg["times"]);calls+=1
                if watch:np.savez_compressed(out/"actual_B1.npz",final=pred.cpu().numpy(),**trace_arrays(trace))
                if mode=="none":assert np.array_equal(pred.cpu().numpy(),baseline[role,k])
                raw=n.unnormalize(pred.clone());p=f(raw,s)[:,:,:22]
                measure(role,arm,mode,k,raw.cpu().numpy(),p.cpu().numpy(),pred.cpu().numpy(),values,items)
                noises.append(dict(role=role,arm=arm,mode=mode,seed_index=k,noise=noise))
            print("R29_SAMPLE",domain,role,mode,flush=True)
    assert digest(d.model)==frozen
    gates=[]
    for role in ("train","holdout"):
        for model in (("B0","B1") if domain=="aist" else ("F0","F1")):
            tests=[]
            for i in range(4):
                for k in range(2):
                    def get(m,mode):return next(x for x in records if (x["role"],x["arm"],x["mode"],x["item"],x["seed_index"])==(role,m,mode,i,k))
                    x=get(model,"correct");w=get(model,"swapped");b=get("baseline","none");er=x["metrics"];wrong=w["metrics"]
                    conditions=dict(body_gain=er["body22_rad"]<=.75*b["metrics"]["body22_rad"],body_reference=er["body22_rad"]<wrong["body22_rad"],
                        local_reference=er["pelvis_wrist_rmse_m"]<wrong["pelvis_wrist_rmse_m"],
                        activity=er["body_activity_ratio"] is not None and .6<=er["body_activity_ratio"]<=1.6,
                        extent=er["arm_range_ratio"] is not None and .6<=er["arm_range_ratio"]<=1.6)
                    q,p=x["quality"],b["quality"]
                    proxy=dict(global_jerk=q["jerk_p95_mps3"]<=1.1*p["jerk_p95_mps3"],core_jerk=q["regions"]["core_interior_unprompted"]["p95"]<=1.1*p["regions"]["core_interior_unprompted"]["p95"],
                        root_speed=q["root_speed_p95_mps"]<=1.1*p["root_speed_p95_mps"],minimum_floor_height=x["floor"]["minimum_height_m"]>=b["floor"]["minimum_height_m"]-.02)
                    clock=None
                    if model==arm:
                        z=get(model,"wrong_clock")["metrics"]
                        clock=dict(body=er["body22_rad"]<z["body22_rad"],local_wrist=er["pelvis_wrist_rmse_m"]<z["pelvis_wrist_rmse_m"])
                    tests.append(dict(item=i,recording=x["recording"],seed_index=k,tests=conditions,source_pass=all(conditions.values()),proxy=proxy,proxy_pass=all(proxy.values()),correct_clock_better=clock))
            windows=sum(all(t["source_pass"] for t in tests if t["item"]==i) for i in range(4))
            groups=sorted({t["recording"] for t in tests})
            passed=sum(all(t["source_pass"] for t in tests if t["recording"]==g) for g in groups)
            gates.append(dict(role=role,arm=model,conditions=tests,windows_passed=windows,recordings=len(groups),recordings_passed=passed,
                source_pass=passed>=3 if domain=="aist" else passed==len(groups),independence_scope="4 exposed AIST recordings" if domain=="aist" else "2 FineDance recordings, 4 dependent windows"))
    result=dict(domain=domain,records=records,gates=gates,noise=noises,sampler_calls=calls,generated_trajectories=4*calls,reused_trajectories=reused,
        optimizer_steps=0,NoRef_original_exact=True,reference_state_write=False,operational_qualified=False,full_native_status="NOT_EVALUABLE",whole_song_generations=0)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,arrays_sha256=sha(path),trace_files={p.name:sha(p) for p in out.glob("actual_*.npz")}))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    a.close();print("R29_SOURCE_GATES",domain,[(x["arm"],x["role"],x["recordings_passed"],x["recordings"]) for x in gates],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--domain",required=True,choices=["aist","fine"]);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.domain,a.replay_from)

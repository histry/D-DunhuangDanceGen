"""Original source and proxy gates, paired physical error and exact frozen R0 reuse."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from .common import values,runtime,checked_parent
from .metrics import measure
from .objective import suppress_direct_root_delta
from dunhuang.research.time_r29.common import packet
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,digest,restore,motion_metrics,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.trace import sample
from dunhuang.research.contract_r28.input_contract import foot_proxy
from dunhuang.research.reference_r26.evaluate import timing

def gates(records,arms,roles):
    result=[]
    def get(role,arm,mode,i,k):
        return next(x for x in records if (x["role"],x["arm"],x["mode"],x["item"],x["seed_index"])==(role,arm,mode,i,k))
    for role in roles:
        for arm in arms:
            tests=[]
            for i in range(4):
                for k in range(2):
                    x=get(role,arm,"correct",i,k);w=get(role,arm,"swapped",i,k);base=get(role,"NoRef","none",i,k);clock=get(role,arm,"wrong_clock",i,k)
                    er=x["metrics"];wr=w["metrics"];br=base["metrics"];q=x["quality"];bq=base["quality"]
                    conditions=dict(body_gain=er["body22_rad"]<=.75*br["body22_rad"],body_reference=er["body22_rad"]<wr["body22_rad"],
                      local_reference=er["pelvis_wrist_rmse_m"]<wr["pelvis_wrist_rmse_m"],activity=er["body_activity_ratio"] is not None and .6<=er["body_activity_ratio"]<=1.6,
                      extent=er["arm_range_ratio"] is not None and .6<=er["arm_range_ratio"]<=1.6)
                    proxy=dict(global_jerk=q["jerk_p95_mps3"]<=1.1*bq["jerk_p95_mps3"],
                      core_jerk=q["regions"]["core_interior_unprompted"]["p95"]<=1.1*bq["regions"]["core_interior_unprompted"]["p95"],
                      root_speed=q["root_speed_p95_mps"]<=1.1*bq["root_speed_p95_mps"],
                      minimum_floor_height=x["floor"]["minimum_height_m"]>=base["floor"]["minimum_height_m"]-.02)
                    tests.append(dict(item=i,case=x["case"],seed_index=k,tests=conditions,source_pass=all(conditions.values()),proxy=proxy,proxy_pass=all(proxy.values()),
                        correct_clock_better=dict(body=er["body22_rad"]<clock["metrics"]["body22_rad"],local_wrist=er["pelvis_wrist_rmse_m"]<clock["metrics"]["pelvis_wrist_rmse_m"])))
            passed=sum(all(t["source_pass"] for t in tests if t["item"]==i) for i in range(4))
            result.append(dict(role=role,arm=arm,conditions=tests,recordings_passed=passed,recordings=4,source_pass=passed>=3,independence="Historical R26 exposed recordings; two seeds not independent sources"))
    return result

def run(cfg,root,out,arms,roles,replay=None,root_diagnostic=False):
    root=Path(root);out=fresh(out);forbid_optimizers()
    d,n,s,f,e,info,a=runtime(cfg);frozen=digest(d.model);arrays={};records=[];noise=[];calls=0;reused=0
    old=np.load(checked_parent(cfg,"evaluation_coverage/arrays.npz"))
    def add(role,arm,mode,k,pred,raw,points,data,items,reuse=False):
        nonlocal reused
        stem=f"{role}_{arm}_{mode}_{k}"
        for key,value in (("normalized",pred),("139",raw),("22",points)):arrays[stem+"_"+key]=value
        for i,row in enumerate(data):
            quality=motion_metrics(raw[i],points[i],np.zeros(256,bool));quality.pop("foot_geometry_proxy",None)
            records.append(dict(role=role,arm=arm,mode=mode,seed_index=k,item=i,case=items[i],reused=reuse,
                **measure(raw[i],points[i],row["model_raw"],row["model_points"]),quality=quality,floor=foot_proxy(points[i]),timing_proxy=timing(raw[i],row["model_raw"])))
        if reuse:reused+=4
    for role in roles:
        data,items=values(cfg,"aist",role)
        for k in range(2):
            for arm,mode,oldmode in [("NoRef","none","none")]+[("R0",m,m) for m in ("correct","wrong_clock","swapped")]:
                stem=f"{role}_COVER_{oldmode}_{k}"
                add(role,arm,mode,k,old[stem+"_normalized"],old[stem+"_139"],old[stem+"_22"],data,items,True)
    for arm in list(arms)+(["RootDelta0"] if root_diagnostic else []):
        model=read(root/f"models/{arm}/result.json") if arm!="RootDelta0" else {"aligned":False}
        a.aligned=model["aligned"];restore(a,root/f"models/{arm}/adapter.pt" if arm!="RootDelta0" else checked_parent(cfg,"models/COVER/adapter.pt"))
        for role in roles:
            data,items=values(cfg,"aist",role);music=torch.tensor(np.stack([x["music"] for x in data]),device="cuda")
            for mode in (("correct",) if arm=="RootDelta0" else ("correct","wrong_clock","swapped","none")):
                for k,seeds in enumerate(read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"]):
                    with suppress_direct_root_delta(d,a) if arm=="RootDelta0" else nullcontext() as counts:
                        pred,nr,_=sample(d,a,None if mode=="none" else packet(data,mode),music,seeds)
                    if mode=="none":assert np.array_equal(pred.cpu().numpy(),old[f"{role}_COVER_none_{k}_normalized"])
                    raw=n.unnormalize(pred.clone())
                    with torch.no_grad():points=f(raw,s)[:,:,:22]
                    add(role,arm,mode,k,pred.cpu().numpy(),raw.cpu().numpy(),points.cpu().numpy(),data,items)
                    noise.append(dict(arm=arm,role=role,mode=mode,seed_index=k,noise=nr,Root_delta_counterfactual=counts));calls+=1
                print("R30_SAMPLE",arm,role,mode,flush=True)
    g=gates(records,["R0"]+list(arms),roles);assert digest(d.model)==frozen;a.close()
    result=dict(records=records,gates=g,noise=noise,sampler_calls=calls,generated_trajectories=4*calls,reused_trajectories=reused,
        optimizer_steps=0,reference_state_write=False,Root_delta_suppression_is_not_independent_NoRef_Root=True,
        full_native_status="NOT_EVALUABLE",whole_song_generations=0,operational_qualified=False)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info,arrays_sha256=sha(path)))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    print("R30_GATES",[(t["arm"],t["role"],t["recordings_passed"]) for t in g],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arms",nargs="*",default=[]);p.add_argument("--roles",nargs="+",default=["train","holdout"],choices=["train","holdout"]);p.add_argument("--root-diagnostic",action="store_true");p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.arms,a.roles,a.replay_from,a.root_diagnostic)

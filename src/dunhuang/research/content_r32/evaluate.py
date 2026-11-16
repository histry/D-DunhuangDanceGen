"""Full sampling, fixed CanonicalIO NoRef, and unchanged source/proxy comparisons."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from dunhuang.research.state_r27.common import write,fresh,commit,forbid_optimizers,same_arrays,motion_metrics
from dunhuang.research.state_r27.trace import sample
from dunhuang.research.time_r29.common import packet
from dunhuang.research.continuity_r30.metrics import measure,geometry,absolute,paired
from dunhuang.research.continuity_r30.evaluate import gates
from dunhuang.research.time_r29.dynamics import decomposition
from dunhuang.research.contract_r28.input_contract import foot_proxy
from dunhuang.research.reference_r26.evaluate import timing
from dunhuang.research.lodge_adapter_r25.sample import sample as change_sample
from dunhuang.research.lodge_adapter_r25.replay import change_conditions

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    enabled=[a for a in ("AIST","ChangE") if read(root/f"models/{a}/result.json")["admitted"]]
    if not enabled:
        write(out/"result.json",dict(admitted=False,optimizer_steps=0,sampler_calls=0,generated_trajectories=0,records=[],operational_qualified=False));return
    pop,ledger,data,z=population(cfg);arrays={};noise=[];calls=0
    old=read(Path(cfg["r31"])/"evaluation/result.json");records=[dict(r,reused=True) for r in old["records"]]
    oldarr=np.load(Path(cfg["r31"])/"evaluation/arrays.npz")
    oldchange=read(Path(cfg["r31"])/"change/result.json")
    changes=[dict(r,reused=True) for r in oldchange["records"]]
    oldchange_arrays=np.load(Path(cfg["r31"])/"change/arrays.npz")
    for arm in enabled:
        d,n,s,f,e,info,b=runtime(cfg,root/f"models/{arm}/adapter.pt");frozen=fingerprint(d,b)
        for role in ("train","holdout"):
            cases=cfg["train_cases"] if role=="train" else cfg["holdout_cases"]
            rows=[data[c] for c in cases];music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
            for mode in ("correct","wrong_clock","swapped","none"):
                for k,seeds in enumerate(read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"]):
                    pred,nr,_=sample(d,b,None if mode=="none" else packet(rows,mode),music,seeds)
                    if mode=="none":assert np.array_equal(pred.cpu().numpy(),oldarr[f"{role}_CanonicalIO_none_{k}_normalized"])
                    raw=n.unnormalize(pred.clone())
                    with torch.no_grad():points=f(raw,s)[:,:,:22]
                    stem=f"{role}_{arm}_{mode}_{k}"
                    for key,x in (("normalized",pred),("139",raw),("22",points)):arrays[stem+"_"+key]=x.cpu().numpy()
                    for i,c in enumerate(cases):
                        row=data[c];q=motion_metrics(raw[i].cpu().numpy(),points[i].cpu().numpy(),np.zeros(256,bool));q.pop("foot_geometry_proxy",None)
                        records.append(dict(role=role,arm=arm,mode=mode,seed_index=k,item=i,case=c,reused=False,
                            **measure(raw[i].cpu().numpy(),points[i].cpu().numpy(),row["model_raw"],row["model_points"]),
                            quality=q,floor=foot_proxy(points[i].cpu().numpy()),timing_proxy=timing(raw[i].cpu().numpy(),row["model_raw"])))
                    noise.append(dict(dataset="aist",arm=arm,role=role,mode=mode,seed_index=k,noise=nr));calls+=1
                print("R32_SAMPLE",arm,role,mode,flush=True)
        conditions=change_conditions();music=torch.tensor(np.stack([z["music"][m] for m,event in conditions]),device="cuda")
        for mode in ("correct","wrong_clock","swapped"):
            events=[1-max(ev,0) if mode=="swapped" else max(ev,0) for m,ev in conditions]
            ref=reference([z["source139"][ev,64:160,7:] for ev in events],drops=[ev<0 for m,ev in conditions]);ref["mapped_source"]=mapping(6,"cuda",wrong=mode=="wrong_clock")
            pred,nr=change_sample(d,b,ref,music,24041,arm);calls+=1
            assert np.array_equal(pred[[0,3]].cpu().numpy(),oldchange_arrays[f"CanonicalIO_{mode}_normalized"][[0,3]])
            raw=n.unnormalize(pred.clone())
            with torch.no_grad():points=f(raw,s)[:,:,:22]
            for key,x in (("normalized",pred),("139",raw),("22",points)):arrays[f"change_{arm}_{mode}_{key}"]=x.cpu().numpy()
            for i,(m,ev) in enumerate(conditions):
                r=raw[i].cpu().numpy();p=points[i].cpu().numpy();q=motion_metrics(r,p,np.zeros(256,bool));q.pop("foot_geometry_proxy",None)
                changes.append(dict(arm=arm,mode=mode,music=m,event=ev,reused=False,errors=[geometry(r,p,a,bp) for a,bp in zip(z["source139"],z["source22"])],
                    generated_absolute=absolute(p),source_absolute=[absolute(v) for v in z["source22"]],
                    paired_root_relative_proxy=[paired(p,v)["Root_relative_body"] for v in z["source22"]],
                    quality=q,floor=foot_proxy(p),dynamics=decomposition(p),Root_truth="UNKNOWN",contact_truth="UNKNOWN",
                    scope="TWO_EXPOSED_TRAINABLE_PRIMITIVES; NOT_NEW_SOURCE_OR_PAIRED_MUSIC_TRUTH"))
            noise.append(dict(dataset="change",arm=arm,mode=mode,noise=nr));print("R32_CHANGE",arm,mode,flush=True)
        assert_frozen(d,b,frozen);b.close()
    own=[];againstR0=[]
    for arm in ("R0","LegacyIO","CanonicalIO",*enabled):
        rows=[r for r in records if r["arm"]==arm]
        denominator=arm if arm in ("R0","LegacyIO","CanonicalIO") else "CanonicalIO"
        own.extend(gates(rows+[dict(r,arm="NoRef") for r in records if r["arm"]==denominator and r["mode"]=="none"],[arm],["train","holdout"]))
        againstR0.extend(gates(rows+[dict(r,arm="NoRef") for r in records if r["arm"]=="R0" and r["mode"]=="none"],[arm],["train","holdout"]))
    result=dict(admitted=True,records=records,change_records=changes,noise=noise,gates_own_NoRef=own,gates_frozen_R0_NoRef=againstR0,
        fixed_Canonical_NoRef_exact=True,optimizer_steps=0,sampler_calls=calls,generated_trajectories=164,reused_trajectories=246,
        operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    assert calls==38
    np.savez_compressed(out/"arrays.npz",**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),arrays_sha256=sha(out/"arrays.npz")))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
    print("R32_EVAL_GATES",[(x["arm"],x["role"],x["recordings_passed"]) for x in own],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

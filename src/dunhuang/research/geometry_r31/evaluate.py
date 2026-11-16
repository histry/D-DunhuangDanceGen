"""Common geometry evaluation, with own-contract and frozen-NoRef gates kept separate."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import sources,parent_config
from .interface import runtime
from dunhuang.research.continuity_r30.metrics import measure
from dunhuang.research.continuity_r30.evaluate import gates
from dunhuang.research.continuity_r30.common import checked_parent
from dunhuang.research.time_r29.common import packet
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,motion_metrics,digest,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.trace import sample,trace_arrays
from dunhuang.research.contract_r28.input_contract import foot_proxy
from dunhuang.research.reference_r26.evaluate import timing

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pop,ledger,_=sources(cfg)
    data={r["case"]:r["data"] for r in pop if r["domain"]=="aist"}
    old=np.load(checked_parent(parent_config(),"evaluation_coverage/arrays.npz"))
    arrays={};records=[];noise=[];calls=0
    def add(role,arm,mode,k,pred,raw,points,cases,reused=False):
        stem=f"{role}_{arm}_{mode}_{k}"
        for key,x in (("normalized",pred),("139",raw),("22",points)):arrays[stem+"_"+key]=x
        for i,c in enumerate(cases):
            row=data[c];q=motion_metrics(raw[i],points[i],np.zeros(256,bool));q.pop("foot_geometry_proxy",None)
            records.append(dict(role=role,arm=arm,mode=mode,seed_index=k,item=i,case=c,reused=reused,
                **measure(raw[i],points[i],row["model_raw"],row["model_points"]),
                quality=q,floor=foot_proxy(points[i]),timing_proxy=timing(raw[i],row["model_raw"])))
    for role in ("train","holdout"):
        cases=cfg["train_cases"] if role=="train" else cfg["holdout_cases"]
        for k in range(2):
            for mode in ("correct","wrong_clock","swapped","none"):
                stem=f"{role}_COVER_{mode}_{k}";add(role,"R0",mode,k,old[stem+"_normalized"],old[stem+"_139"],old[stem+"_22"],cases,True)
    for arm in ("LegacyIO","CanonicalIO"):
        d,n,s,f,e,info,b=runtime(cfg,arm=="CanonicalIO",root/f"models/{arm}/adapter.pt");frozen=digest(d.model)
        for role in ("train","holdout"):
            cases=cfg["train_cases"] if role=="train" else cfg["holdout_cases"];rows=[data[c] for c in cases]
            music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
            for mode in ("correct","wrong_clock","swapped","none"):
                for k,seeds in enumerate(read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"]):
                    trace=role=="train" and mode=="correct" and k==0
                    pred,nr,state=sample(d,b,None if mode=="none" else packet(rows,mode),music,seeds,trace=trace,selected=cfg["times"])
                    if trace:np.savez_compressed(out/f"actual_{arm}.npz",final=pred.cpu().numpy(),**trace_arrays(state))
                    raw=n.unnormalize(pred.clone())
                    with torch.no_grad():points=f(raw,s)[:,:,:22]
                    add(role,arm,mode,k,pred.cpu().numpy(),raw.cpu().numpy(),points.cpu().numpy(),cases)
                    noise.append(dict(role=role,arm=arm,mode=mode,seed_index=k,noise=nr));calls+=1
                print("R31_SAMPLE",arm,role,mode,flush=True)
        assert digest(d.model)==frozen;b.close()
    own=[];frozen_gates=[]
    for arm in ("R0","LegacyIO","CanonicalIO"):
        rows=[r for r in records if r["arm"]==arm]
        own.extend(gates(rows+[dict(r,arm="NoRef") for r in rows if r["mode"]=="none"],[arm],["train","holdout"]))
        frozen_gates.extend(gates(rows+[dict(r,arm="NoRef") for r in records if r["arm"]=="R0" and r["mode"]=="none"],[arm],["train","holdout"]))
    result=dict(records=records,gates_own_NoRef=own,gates_frozen_NoRef=frozen_gates,noise=noise,
        sampler_calls=calls,generated_trajectories=4*calls,reused_trajectories=64,optimizer_steps=0,
        new_contract_qualification_not_inherited=True,reference_state_write=False,operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    np.savez_compressed(out/"arrays.npz",**arrays);write(out/"result.json",result)
    write(out/"protocol.json",dict(config=cfg,code_commit=commit(),files={p.name:sha(p) for p in out.glob("*.npz")}))
    if replay:
        assert result==read(Path(replay)/"result.json")
        for p in out.glob("*.npz"):same_arrays(p,Path(replay)/p.name)
    print("R31_SOURCE_GATES",[(x["arm"],x["role"],x["recordings_passed"]) for x in own],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

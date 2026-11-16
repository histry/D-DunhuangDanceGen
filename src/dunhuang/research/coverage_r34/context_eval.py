"""Actual soft context interventions; evaluate unoverridden gap and both raw boundaries."""
import argparse
from .common import *
from .context_contract import context
from .context_train import restore_context,cg_values
from .data import packet
from .metrics import measure
from dunhuang.research.state_r27.common import metrics
from dunhuang.research.contract_r28.input_contract import foot_proxy

def paired_metrics(raw,p,truth,target,gap=(64,160)):
    a,b=gap;mask=np.zeros(256,bool);mask[a:b]=True
    rel=p-p[:,:1];source_rel=target-target[:,:1];e=rel-source_rel
    errors={}
    for boundary in gap:
        for k in (1,2,3):
            starts=np.arange(256-k);select=(starts<boundary)&(starts+k>=boundary)
            errors[f"{boundary}_D{k}"]=float(np.sqrt(np.mean((np.diff(e,n=k,axis=0)[select]*30**k)**2)))
    return dict(geometry=metrics(raw,p,truth,target,mask=mask),
        root_rmse_m=float(np.sqrt(np.mean((raw[a:b,4:7]-truth[a:b,4:7])**2))),cross=errors,
        gap_body_jerk=float(np.percentile(np.linalg.norm(np.diff(rel[a:b],n=3,axis=0)*30**3,axis=-1),95)),
        floor_proxy=foot_proxy(p),root_truth="OBSERVED_COMPLETE_AIST_COMMON_MODEL0",contact_truth="HEIGHT_PROXY_NOT_MEASURED")

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    trained=read(root/"context/result.json")
    if not trained["admitted"]:
        write(out/"result.json",dict(admitted=False,reason=trained["reason"],optimizer_steps=0,sampler_calls=0,generated_trajectories=0,operational_qualified=False,whole_song_generations=0));return
    rt=Runtime(cfg,root/"models/Coverage/adapter.pt",context_training=True);restore_context(rt,root/"context/context.pt")
    _,_,data,z=population_data();cfg32=read("src/dunhuang/research/content_r32/config.json");arrays={};records=[];noise=[];calls=0
    seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0];baseline=np.load("outputs/geometry_r31_02/evaluation/arrays.npz")
    settings=[("none",m) for m in ("none","both","no_left","no_right","wrong","partial")]+[("correct",m) for m in ("both","wrong","none")]+[("swapped","both")]
    for role in ("train","holdout"):
        cases=cfg32["train_cases"] if role=="train" else cfg32["holdout_cases"];rows=[data[c] for c in cases]
        y=rt.n.normalize(torch.tensor(np.stack([r["model_raw"] for r in rows]),device="cuda"));music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
        for src,mode in settings:
            donors=rows if src!="swapped" else rows[1:]+rows[:1];donor_cases=cases if src!="swapped" else cases[1:]+cases[:1]
            ref=None if src=="none" else source([core_body(r) for r in donors],"aist",uids=[f"aist_{c}" for c in donor_cases])
            cp=None if mode=="none" else context(y,[64,160],mode,cfg["context_seed"])
            pred,nr,_=sample(rt,music,source_packet=ref,context_packet=cp,seeds=seeds);noise.append(nr);calls+=1
            if mode=="none":assert np.array_equal(pred.cpu().numpy(),baseline[f"{role}_CanonicalIO_{'none' if src=='none' else 'correct'}_0_normalized"])
            raw=rt.n.unnormalize(pred.clone())
            with torch.no_grad():p=rt.f(raw,rt.s)[:,:,:22]
            raw=raw.cpu().numpy();p=p.cpu().numpy();key=f"{role}_{src}_{mode}"
            arrays[key+"_normalized"]=pred.cpu().numpy();arrays[key+"_139"]=raw;arrays[key+"_22"]=p
            for i,c in enumerate(cases):records.append(dict(role=role,source=src,context=mode,case=c,item=i,**paired_metrics(raw[i],p[i],data[c]["model_raw"],data[c]["model_points"])))
            print("R34_CONTEXT_SAMPLE",role,src,mode,flush=True)
    a=dict(np.load(root/"data/arrays.npz"));conditions=[(m,e) for m in range(2) for e in (-1,0,1)];music=torch.tensor(np.stack([z["music"][m] for m,e in conditions]),device="cuda")
    change=[];expected=np.load(root/"evaluation/arrays.npz")
    for src,mode in [("correct",m) for m in ("none","both","no_left","no_right","wrong")]+[("swapped","both")]:
        cp=None if mode=="none" else cg_values(a,conditions)
        if cp is not None:
            if mode=="no_left":cp.valid[:,:8]=False
            elif mode=="no_right":cp.valid[:,8:]=False
            elif mode=="wrong":
                # Swap only the two actual source bodies, not adjacent batch/music identities.
                cp=cg_values(dict(a,body=a["body"][[1,0]]),conditions)
        pred,nr,_=sample(rt,music,source_packet=packet(a,conditions,True,src),context_packet=cp,common_seed=24041);calls+=1;noise.append(nr)
        assert np.array_equal(pred[[0,3]].cpu().numpy(),expected["Coverage_correct_normalized"][[0,3]])
        if mode=="none":assert np.array_equal(pred.cpu().numpy(),expected["Coverage_correct_normalized"])
        raw=rt.n.unnormalize(pred.clone())
        with torch.no_grad():p=rt.f(raw,rt.s)[:,:,:22]
        raw=raw.cpu().numpy();p=p.cpu().numpy();key=f"change_{src}_{mode}"
        arrays[key+"_normalized"]=pred.cpu().numpy();arrays[key+"_139"]=raw;arrays[key+"_22"]=p
        for i,(m,e) in enumerate(conditions):change.append(dict(source=src,context=mode,music=m,event=e,errors=[measure(raw[i],p[i],body,points) for body,points in zip(a["body"],a["local_points"])]))
        print("R34_CONTEXT_SAMPLE","change",src,mode,flush=True)
    def avg(mode,key):
        rows=[r for r in records if r["role"]=="holdout" and r["source"]=="none" and r["context"]==mode]
        return float(np.mean([r["geometry"]["body22_rad"] if key=="body" else r["root_rmse_m"] if key=="root" else np.mean([r["cross"]["64_D2"],r["cross"]["160_D2"]]) for r in rows]))
    body={m:avg(m,"body") for m in ("none","both","no_left","no_right","wrong","partial")}
    cross={m:avg(m,"cross") for m in body};roots={m:avg(m,"root") for m in body}
    gate=dict(holdout_body=body,holdout_cross_D2=cross,holdout_root_rmse=roots,
        context_local_pass=body["both"]<min(body["none"],body["no_left"],body["no_right"]) and cross["both"]<cross["none"] and roots["both"]<roots["none"],
        gate_scope="FINITE_EXPOSED_FOUR_RECORDINGS_NOT_POPULATION_SIGNIFICANCE",operational_qualified=False)
    rt.check();rt.close();assert calls==26
    result=dict(admitted=True,records=records,change_records=change,gate=gate,noise=noise,optimizer_steps=0,sampler_calls=calls,generated_trajectories=116,
        hard_observation_override=False,target_gap_values_in_context=False,general_and_source_only_exact=True,full_native_status="NOT_EVALUABLE_22_OF24",
        operational_qualified=False,whole_song_generations=0)
    np.savez_compressed(out/"arrays.npz",**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(out/"arrays.npz")))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
    print("R34_CONTEXT_GATE",gate,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

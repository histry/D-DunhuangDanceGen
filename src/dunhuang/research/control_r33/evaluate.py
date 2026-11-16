"""Frozen comparison on full DDIM outputs; synthetic gauges never become Root truth."""
import argparse
import numpy as np
import torch
from .common import *
from .runtime import Runtime,restore_change
from dunhuang.research.state_r27.common import metrics,motion_metrics
from dunhuang.research.time_r29.dynamics import decomposition
from dunhuang.research.contract_r28.input_contract import foot_proxy
from dunhuang.research.lodge_adapter_r25.model import PARTS

def source_metrics(raw,points,body,target):
    p=np.asarray(points[64:160],dtype=np.float64);p=p-p[:,:1]
    target=np.asarray(target,dtype=np.float64);target=target-target[:,:1]
    truth=np.zeros((96,139),np.float32);truth[:,7:]=body
    geometry=metrics(raw[64:160],points[64:160],truth,target,mask=np.ones(96,bool))
    e=p-target
    derivatives={str(k):dict(rmse=float(np.sqrt(np.mean((np.diff(e,n=k,axis=0)*30**k)**2))),
        p95=float(np.percentile(np.linalg.norm(np.diff(e,n=k,axis=0)*30**k,axis=-1),95)),templates=96-k) for k in (1,2,3)}
    groups={str(i):{str(k):float(np.sqrt(np.mean((np.diff(e[:,group],n=k,axis=0)*30**k)**2))) for k in (1,2,3)} for i,group in enumerate(PARTS)}
    spectrum=np.fft.rfft((e-e.mean(0))*np.hanning(96)[:,None,None],axis=0);freq=np.fft.rfftfreq(96,1/30)
    power=(abs(spectrum)**2).mean(axis=(1,2))
    edge={str(t):dict(position_step_m=float(np.linalg.norm(points[t]-points[t-1],axis=-1).mean()),
        velocity_change_mps=float(np.linalg.norm((points[t+1]-2*points[t]+points[t-1])*30,axis=-1).mean())) for t in (64,160)}
    return dict(geometry=geometry,paired_body=derivatives,body_groups=groups,
        high_frequency_residual_power_ge6Hz=float(power[freq>=6].sum()),spectrum_scope="HANN96_ROOT_RELATIVE_POSITION_ERROR; NOT_SEMANTIC_TRUTH",
        dynamics=decomposition(points),edges=edge,Root_truth="UNKNOWN",contact_truth="UNKNOWN")

def run(cfg,root,out,replay=None):
    out=fresh(out);root=Path(root);forbid_optimizers();_,_,data,z=population_data();rt=Runtime(cfg)
    initial={k:v.detach().clone() for k,v in rt.c.change.state_dict().items()}
    held=np.load(root/"held_compilation_v2/arrays.npz");heldmeta=read(root/"held_compilation_v2/result.json")
    bodies={"train":z["source139"][:,64:160,7:],"held":held["body"]}
    points={"train":z["source22"][:,64:160],"held":held["local_points"]}
    ids={"train":["female_FeiTian","male_pipa_1"],"held":[r["source_uid"] for r in heldmeta["records"]]}
    arrays={};records=[];noise=[];retention=[];music_response=[];calls=0;conditions=[(m,e) for m in range(2) for e in (-1,0,1)]
    basearr=np.load(Path(cfg["r32"])/"evaluation/arrays.npz")
    generalarr=np.load(Path(cfg["r31"])/"evaluation/arrays.npz");cfg32=read("src/dunhuang/research/content_r32/config.json")
    for arm in ("Frozen","Extra","Dyn"):
        if arm=="Frozen":rt.c.change.load_state_dict(initial)
        else:restore_change(rt,root/f"models/{arm}/adapter.pt")
        for role in ("train","holdout"):
            cases=cfg32[role+"_cases"] if role=="train" else cfg32["holdout_cases"]
            rows=[data[c] for c in cases];music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
            seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0]
            for mode in ("correct","none"):
                packet=None if mode=="none" else source([core_body(r) for r in rows],"aist",uids=[f"aist_{c}" for c in cases])
                pred,nr,_=sample(rt,music,source_packet=packet,seeds=seeds);calls+=1
                expected=generalarr[f"{role}_CanonicalIO_{mode}_0_normalized"];assert np.array_equal(pred.cpu().numpy(),expected)
                arrays[f"aist_{arm}_{role}_{mode}"]=pred.cpu().numpy()
                retention.append(dict(arm=arm,role=role,mode=mode,trajectories=4,exact_R31=True))
                noise.append(nr)
        for role in ("train","held"):
            music=torch.tensor(np.stack([z["music"][m] for m,e in conditions]),device="cuda")
            for mode in ("correct","wrong_clock","swapped"):
                # Swap each real reference to the other identified source; NoRef donor stays masked.
                selected=[1-max(e,0) if mode=="swapped" else max(e,0) for m,e in conditions]
                packet=source([bodies[role][j] for j in selected],"change",mode="wrong_clock" if mode=="wrong_clock" else "correct",
                    uids=[ids[role][j] for j in selected],drops=[e<0 for m,e in conditions])
                pred,nr,_=sample(rt,music,source_packet=packet,common_seed=24041);calls+=1;noise.append(nr)
                normalized=pred.cpu().numpy();assert np.array_equal(normalized[[0,3]],basearr["change_ChangE_correct_normalized"][[0,3]])
                if arm=="Frozen" and role=="train":assert np.array_equal(normalized,basearr[f"change_ChangE_{mode}_normalized"])
                raw=rt.n.unnormalize(pred.clone())
                with torch.no_grad():p=rt.f(raw,rt.s)[:,:,:22]
                raw=raw.cpu().numpy();p=p.cpu().numpy();stem=f"{role}_{arm}_{mode}"
                arrays[stem+"_normalized"]=normalized;arrays[stem+"_139"]=raw;arrays[stem+"_22"]=p
                for i,(m,e) in enumerate(conditions):
                    q=motion_metrics(raw[i],p[i],np.zeros(256,bool));q.pop("foot_geometry_proxy",None)
                    records.append(dict(arm=arm,role=role,mode=mode,music=m,event=e,source_uid=None if e<0 else ids[role][e],
                        errors=[source_metrics(raw[i],p[i],body,target) for body,target in zip(bodies[role],points[role])],
                        quality=q,floor=foot_proxy(p[i])))
                if mode=="correct":
                    rel=p[:,:,]-p[:,:,:1]
                    music_response.extend(dict(arm=arm,role=role,event=e,root_relative_rms_m=float(np.sqrt(np.mean((rel[j+3,64:160]-rel[j,64:160])**2))),
                        scope="SAME_NOISE_DIFFERENT_MUSIC_RESPONSE_NOT_COORDINATION_TRUTH") for j,e in enumerate((-1,0,1)))
                print("R33_SAMPLE",arm,role,mode,flush=True)
        rt.check()
    rt.close()
    def avg(arm,event,path):
        rows=[r for r in records if r["arm"]==arm and r["role"]=="train" and r["mode"]=="correct" and r["event"]==event]
        def get(r):
            x=r["errors"][event]
            for k in path:x=x[k]
            return x
        return float(np.mean([get(r) for r in rows]))
    gates=[]
    for arm in ("Extra","Dyn"):
        source_gates=[]
        for e in (0,1):
            angle=avg(arm,e,["geometry","body22_rad"])/avg("Frozen",e,["geometry","body22_rad"])
            wrist=avg(arm,e,["geometry","world_wrist_rmse_m"])/avg("Frozen",e,["geometry","world_wrist_rmse_m"])
            activity=avg(arm,e,["geometry","body_activity_ratio"])/avg("Frozen",e,["geometry","body_activity_ratio"])
            extent=avg(arm,e,["geometry","arm_range_ratio"])/avg("Frozen",e,["geometry","arm_range_ratio"])
            source_gates.append(dict(event=e,angle_ratio=angle,wrist_ratio=wrist,activity_ratio=activity,extent_ratio=extent,
                passed=angle<=1.1 and wrist<=1.1 and .9<=activity<=1.1 and .9<=extent<=1.1))
        d2=avg(arm,1,["paired_body","2","rmse"])/avg("Frozen",1,["paired_body","2","rmse"])
        d3=avg(arm,1,["paired_body","3","rmse"])/avg("Frozen",1,["paired_body","3","rmse"])
        jerk=avg(arm,1,["dynamics","regions","core_interior","Root_relative_body"])/avg("Frozen",1,["dynamics","regions","core_interior","Root_relative_body"])
        feid2=avg(arm,0,["paired_body","2","rmse"])/avg("Frozen",0,["paired_body","2","rmse"])
        feijerk=avg(arm,0,["dynamics","regions","core_interior","Root_relative_body"])/avg("Frozen",0,["dynamics","regions","core_interior","Root_relative_body"])
        gates.append(dict(arm=arm,source_retention=source_gates,pipa_D2_ratio=d2,pipa_D3_ratio=d3,pipa_body_jerk_ratio=jerk,
            feitian_D2_ratio=feid2,feitian_body_jerk_ratio=feijerk,
            local_dynamics_package_pass=all(r["passed"] for r in source_gates) and max(d2,d3,jerk)<1 and max(feid2,feijerk)<=1.1,
            full_physical_qualified=False))
    result=dict(records=records,retention=retention,gates=gates,music_response=music_response,noise=noise,
        optimizer_steps=0,sampler_calls=calls,generated_trajectories=156,general_retention_exact=True,
        NoRef_exact=True,held_records=heldmeta["records"],held_used_for_training_or_selection=False,
        fixed_single_seed_research=True,human_reviewed=False,operational_qualified=False,full_native_status="NOT_EVALUABLE_22_OF24",whole_song_generations=0)
    assert calls==30
    np.savez_compressed(out/"arrays.npz",**arrays);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(out/"arrays.npz"),
        held_arrays_sha256=sha(root/"held_compilation_v2/arrays.npz"),retention_gates="Registered before evaluation: angle/wrist<=1.1; activity/extent0.9..1.1; pipaD2/D3/jerk<1; feitianD2/jerk<=1.1. Not Native admission"))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
    print("R33_EVAL_DONE",[(r["arm"],r["local_dynamics_package_pass"]) for r in gates],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

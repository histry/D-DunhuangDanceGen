"""Task-specific physical readouts on all global frames, independent of training loss."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.continuous_r16.evaluate import seam,angular
from dunhuang.research.masked_denoiser.metrics import decode
from dunhuang.research.boundary_state_r14.bounds import immutable_jerk_bound

def score(x,target,fk,target_fk,keys,observed_mask=None):
    jerk=np.linalg.norm(np.diff(fk,n=3,axis=0)*30.**3,axis=-1)
    edge=np.r_[0:8,len(x)-8:len(x)]
    return dict(frames=len(x),root_rmse_m=float(np.sqrt(np.mean((x[:,4:7]-target[:,4:7])**2))),
        pose_error_rad=angular(x,target),foot_rmse_m=float(np.sqrt(np.mean((fk[:,[7,8,10,11]]-target_fk[:,[7,8,10,11]])**2))),
        jerk_max_mps3=float(jerk.max()),jerk_p95_mps3=float(np.percentile(jerk,95)),
        source_key_pose_rad=angular(x[keys],target[keys],list(range(12,20))),
        edge_root_rmse_m=float(np.sqrt(np.mean((x[edge,4:7]-target[edge,4:7])**2))),
        activity_mps=float(np.linalg.norm(np.diff(fk,axis=0)*30,axis=-1).mean()),
        root_velocity_error_mps=float(np.sqrt(np.mean((np.diff(x[:,4:7],axis=0)-np.diff(target[:,4:7],axis=0))**2)))*30,
        root_acceleration_error_mps2=float(np.sqrt(np.mean((np.diff(x[:,4:7],n=2,axis=0)-np.diff(target[:,4:7],n=2,axis=0))**2)))*900)

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pre=read(root/"data/protocol.json");native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from contracts.physical_quality import PhysicalQualityLimits
    scaler=np.load(checked(pre["assets"]["scaler"]));data=read(root/"data/result.json")
    rows={r["case"]:r for r in data["records"] if r["role"]=="eval"};factor=[];music=[];full=[]
    audit=read(root/"audit/result.json");roll=read(root/"rollout/result.json");saved={}
    def evaluate(binding,kind):
        case=binding["case"];z=np.load(checked(rows[case]["arrays"]));trace=np.load(checked(binding["trace"]))
        target=z["native"][:len(trace["sample"])];x=decode(trace["sample"],scaler,target)
        fk,physical=audit_native(x);gtfk=fk_24_np(target)
        folder=out/(kind+"_"+str(len(saved)));folder.mkdir()
        np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"physical.json",physical)
        known=trace["known"];root_known=known[:,4:7];raw=decode(trace["raw_final"],scaler,target)
        common=dict(case=case,kind=kind,source_frame_start=rows[case]["left"]["start_frame"],native_ok=physical["absolute"]["ok"],
            raw_known_root_rmse_m=float(np.sqrt(np.mean((raw[:,4:7]-target[:,4:7])[root_known]**2))) if root_known.any() else None,
            contact_truth="UNKNOWN")
        if kind=="factor":
            old=read(Path(cfg["r16_root"])/"preflight/protocol.json")["pairs"][case];oldz=np.load(checked(old["arrays"]))
            keys=np.unique(np.concatenate([np.arange(t-1,t+2) for t in oldz["keyframes"]]))
            result=dict(common,position=binding["position"],hard=binding["hard"],event=binding["event"],
                **score(x,target,fk,gtfk,keys),joins=[seam(x,fk,t) for t in (8,52,60,75,90,98,142)])
            halo=z["halo150"].copy()
            if len(halo)!=166:raise ValueError("Real evaluation halo required for factor task")
            halo[8:158]=x;hfk,hphys=audit_native(halo)
            np.save(folder/"halo.npy",halo);np.save(folder/"halo_fk.npy",hfk);write(folder/"halo_physical.json",hphys)
            result.update(external_joins=[seam(halo,hfk,t) for t in (8,158)],halo_native_ok=hphys["absolute"]["ok"])
        else:
            keys=np.r_[55:58,167:170]
            result=dict(common,**score(x,target,fk,gtfk,keys),joins=[seam(x,fk,t) for t in (8,75,112,150,217)],
                real_external_motion_used=False)
            if kind=="music":result["policy"]=binding["policy"]
            else:
                result.update(arm=binding["arm"],task=binding["task"],context_controls=[])
                for timestep in (519,19):
                    for control in ("correct","reversed","removed"):
                        p=trace[str(timestep)+"_"+control]
                        result["context_controls"].append(dict(t=timestep,control=control,
                            reconstruction=float(np.mean((p[:,4:]-z["encoded"][:,4:])**2))))
                if binding["task"]=="progressive":
                    receipt=read(checked(binding["sampler"]));prefix=np.load(checked(receipt["prefix"]))["sample"][:75]
                    assert np.array_equal(trace["sample"][:75,4:],prefix[:,4:])
                    result["generated_prefix_preserved"]=True
                    result["fixed_history_jerk_bound"]=immutable_jerk_bound(fk,PhysicalQualityLimits.from_environment().joint_jerk_mps3_max,edit=(75,225))
        saved[str(folder)]=dict(native=folder.name,kind=kind)
        if replay:
            other=Path(replay)/folder.name
            for file in folder.iterdir():
                if file.suffix==".npy":assert np.array_equal(np.load(file),np.load(other/file.name))
                else:assert read(file)==read(other/file.name)
        return result
    for b in audit["factors"]:factor.append(evaluate(b,"factor"))
    for b in audit["music"]:music.append(evaluate(b,"music"))
    for b in roll["bindings"]:full.append(evaluate(b,"rollout"))
    response=[]
    for i in range(4):
        own=next(m for m in music if m["case"]==i and m["policy"]=="owned")
        independent=next(m for m in music if m["case"]==i and m["policy"]=="independent")
        a=next(b for b in audit["music"] if b["case"]==i and b["policy"]=="owned")
        b=next(b for b in audit["music"] if b["case"]==i and b["policy"]=="independent")
        z=np.load(checked(rows[i]["arrays"]));x=decode(np.load(checked(a["trace"]))["sample"],scaler,z["native"])
        y=decode(np.load(checked(b["trace"]))["sample"],scaler,z["native"])
        response.append(dict(case=i,pose_response_rad=angular(x,y),root_response_rmse_m=float(np.sqrt(np.mean((x[:,4:7]-y[:,4:7])**2)))))
    by={(r["case"],r["arm"],r["task"]):r for r in full};g=cfg["qualification"];cases=[]
    for i in range(4):
        conditions=[]
        for task in ("free","inbetween"):
            b=by[i,"B_cross",task]
            for arm in ("A_numeric","C_self"):
                a=by[i,arm,task]
                conditions.extend(b[key]<=a[key]*(1-g["relative_gain"]) for key in ("root_rmse_m","foot_rmse_m","jerk_max_mps3"))
            means={ctrl:float(np.mean([r["reconstruction"] for r in b["context_controls"] if r["control"]==ctrl])) for ctrl in ("correct","reversed","removed")}
            conditions.append(means["correct"]<means["reversed"] and means["correct"]<means["removed"])
        cases.append(dict(case=i,qualified=all(conditions)))
    result=dict(factors=factor,music=music,music_response=response,rollouts=full,
        qualification=dict(cases=cases,qualified_cases=sum(r["qualified"] for r in cases),
            mechanism_qualified=sum(r["qualified"] for r in cases)>=g["minimum_wins"],operational_qualified=False),
        optimizer_steps=0,new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,saved=saved,
        free_scope="No future external full motion or halo acceptance imposed",
        inbetween_scope="Known boundary, inner8/217, shared75/150 and global worst case all retained",
        progressive_scope="A_numeric generated prefix, not real future motion; not a whole-song execution"))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R17_MECHANISM",result["qualification"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

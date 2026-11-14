"""Native unchanged; separate unknown-boundary compatibility, activity and paired diagnostics."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write
from dunhuang.contracts import matrices
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.continuous_r16.evaluate import seam,angular
from dunhuang.research.masked_denoiser.metrics import decode
from dunhuang.research.global_clock_r17.evaluate import score
from dunhuang.research.boundary_state_r14.bounds import immutable_jerk_bound
from .engine import population

def dynamics(native,fk):
    speed=np.linalg.norm(np.diff(fk,axis=0)*30,axis=-1)
    R=matrices(native[:,7:].reshape(-1,24,6));delta=R[:-1].swapaxes(-1,-2)@R[1:]
    omega=Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude().reshape(-1,24)*30
    relative=R[0].swapaxes(-1,-2)[None]@R
    rom=Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(-1,24)
    hold=speed.mean(-1)<.05;lengths=[];n=0
    for v in np.r_[hold,False]:
        if v:n+=1
        elif n:lengths.append(n);n=0
    return dict(speed_quantiles_mps=np.percentile(speed,[5,50,95]).tolist(),
        angular_speed_quantiles_radps=np.percentile(omega,[5,50,95]).tolist(),
        joint_range_p95_rad=np.percentile(rom,95,axis=0).tolist(),hold_fraction=float(hold.mean()),
        holds_at_least_6_frames=sum(l>=6 for l in lengths),longest_hold_frames=max(lengths,default=0))

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pre,fit,ev=population(cfg);native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from contracts.physical_quality import PhysicalQualityLimits
    scaler=np.load(checked(pre["assets"]["scaler"]));rows={(r["role"],r["case"]):r for r in fit+ev};records=[];saved={}
    for phase in ["tiny"]+(["main"] if (root/"main_rollout").exists() else []):
        for index,b in enumerate(read(root/(phase+"_rollout")/"result.json")["bindings"]):
            row=rows[b["role"],b["case"]];z=np.load(checked(row["arrays"]));tr=np.load(checked(b["trace"]))
            x=decode(tr["sample"],scaler,z["native"]);raw=decode(tr["raw_final"],scaler,z["native"])
            fk,physical=audit_native(x);gtfk=fk_24_np(z["native"]);rawfk=fk_24_np(raw)
            folder=out/(phase+"_"+str(index));folder.mkdir()
            np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"physical.json",physical)
            known=tr["known"];unknown=~known[:,4:];rk=known[:,4:7]
            result=dict(phase=phase,case=b["case"],role=b["role"],arm=b["arm"],task=b["task"],control=b["control"],
                native_ok=physical["absolute"]["ok"],contact_truth="UNKNOWN",
                **score(x,z["native"],fk,gtfk,np.r_[55:58,167:170]),dynamics=dynamics(x,fk),source_dynamics=dynamics(z["native"],gtfk),
                raw_unknown_mse=float(np.mean((tr["raw_final"][:,4:]-z["encoded"][:,4:])[unknown]**2)),
                raw_known_root_rmse_m=float(np.sqrt(np.mean(((tr["raw_final"][:,4:7]-tr["condition"][:,4:7])/scaler["scale"][4:7])[rk]**2))) if rk.any() else None,
                joins=[seam(x,fk,t) for t in (8,75,112,150,217)],raw_joins=[seam(raw,rawfk,t) for t in (8,75,112,150,217)],
                phase_pose_errors_rad=[angular(x[a:b],z["native"][a:b],list(range(12,20))) for a,b in ((0,112),(112,225))],
                context_controls=[])
            for t in (519,19):
                for control in cfg["controls"]:
                    q=tr[str(t)+"_"+control]
                    result["context_controls"].append(dict(t=t,control=control,unknown_mse=float(np.mean((q[:,4:]-z["encoded"][:,4:])[unknown]**2))))
            if b["task"] in ("inbetween","progressive"):
                edit=(8,217) if b["task"]=="inbetween" else (75,225)
                result["immutable_history_bound"]=immutable_jerk_bound(fk,PhysicalQualityLimits.from_environment().joint_jerk_mps3_max,edit=edit)
            if b["task"]=="progressive":
                receipt=read(checked(b["sampler"]));prefix=np.load(checked(receipt["prefix"]))["sample"][:75]
                assert np.array_equal(tr["sample"][:75,4:],prefix[:,4:])
                # Diagnose newly emitted motion separately from immutable-prefix violations.
                result["new_motion_max_jerk_mps3"]=float(np.linalg.norm(np.diff(fk[75:],n=3,axis=0)*30**3,axis=-1).max())
                result["generated_prefix_preserved"]=True
            records.append(result);saved[folder.name]=b
            if replay:
                other=Path(replay)/folder.name
                for f in folder.iterdir():
                    if f.suffix==".npy":assert np.array_equal(np.load(f),np.load(other/f.name))
                    else:assert read(f)==read(other/f.name)
    # Population qualification is separate from tiny overfit admission.
    cases=[]
    for i in range(4) if (root/"main_rollout").exists() else []:
        checks=[]
        for task in ("free","inbetween"):
            def get(arm):return next(r for r in records if r["phase"]=="main" and r["case"]==i and r["task"]==task and r["arm"]==arm and r["control"]=="correct")
            b=get("D2")
            for arm in ("D0","D1"):
                a=get(arm)
                checks += [b[k]<=a[k]*(1-cfg["qualification"]["relative_gain"]) for k in ("foot_rmse_m","jerk_max_mps3")]
                checks.append(b["activity_mps"]>=a["activity_mps"]*cfg["qualification"]["minimum_activity_ratio"])
            cf={c:np.mean([v["unknown_mse"] for v in b["context_controls"] if v["control"]==c]) for c in ("correct","content","removed")}
            checks.append(cf["correct"]<min(cf["content"],cf["removed"]))
        cases.append(dict(case=i,qualified=bool(all(checks))))
    result=dict(records=records,qualification=dict(cases=cases,qualified_cases=sum(r["qualified"] for r in cases),
        mechanism_qualified=bool(cases) and sum(r["qualified"] for r in cases)>=cfg["qualification"]["minimum_cases"],
        tiny_is_generalization=False),optimizer_steps=0,D3_admitted=False,new_whole_song_generations=0,operational_qualified=False)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,saved=saved,
        free_root_rmse_scope="Paired reconstruction diagnostic, not unique artistic target",
        amplitude_scope="Quantiles,range,holds; activity gate prevents accepting only motion shrinkage"))
    print("R18_NATIVE",len(records),result["qualification"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

"""Unchanged Native; fixed/crossing/editable domains computed from actual masks."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.masked_denoiser.metrics import decode
from dunhuang.research.global_clock_r17.evaluate import score
from dunhuang.research.continuous_r16.evaluate import angular,seam
from dunhuang.research.task_denoising_r18.evaluate import dynamics
from dunhuang.research.objective_r19.readouts import balanced
from dunhuang.research.boundary_state_r14.bounds import immutable_jerk_bound
from .contract import numpy_regions

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pre=read(root/"data/protocol.json");native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    scaler=np.load(checked(pre["assets"]["scaler"]));population=read(root/"data/result.json");by={r["case"]:r for r in population["records"]}
    records=[];saved={}
    for index,b in enumerate(read(root/"rollout/result.json")["bindings"]):
        z=np.load(checked(b["data"]));tr=np.load(checked(b["trace"]));target=z["native"];gt=fk_24_np(target)
        x=decode(tr["sample"],scaler,target);raw=decode(tr["raw_final"],scaler,target);fk,physical=audit_native(x)
        folder=out/str(index);folder.mkdir();np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"physical.json",physical)
        left,right=cfg["contexts"][b["variant"]];end=225-right
        original=tr["known"].copy();original[:left,4:]=True;original[end:,4:]=True
        errors,counts=balanced(tr["raw_final"],z["encoded"],original)
        jerk=np.linalg.norm(np.diff(fk,n=3,axis=0)*30**3,axis=-1);domains={}
        for name,mask in numpy_regions(tr["hard"][None],3).items():
            v=jerk[mask];domains[name]=dict(templates=int(mask.sum()),max=float(v.max()) if v.size else None,p95=float(np.percentile(v,95)) if v.size else None)
        # Preserve the historical R20 measurement as a distinct readout.
        starts=np.arange(222);legacy=np.ones(222,bool)
        for cut in (8,75,112,150,217):legacy&=~((starts<cut)&(starts+3>=cut))
        keys=np.r_[left+(end-left)//4-1:left+(end-left)//4+2,left+3*(end-left)//4-1:left+3*(end-left)//4+2]
        row=dict(arm=b["arm"],case=b["case"],variant=b["variant"],control=b["control"],role=b["role"],source_group=b["source_group"],
            native_ok=physical["absolute"]["ok"],native_reasons=physical["absolute"]["reasons"],native_audit=physical["absolute"]["audit"],
            **score(x,target,fk,gt,keys),raw_losses=errors,counts=counts,hard_domains=domains,
            legacy_r20_interior=dict(templates=int(legacy.sum()),max=float(jerk[legacy].max()),p95=float(np.percentile(jerk[legacy],95))),
            raw_observed_pose_error_rad=angular(raw[np.r_[0:left,end:225]],target[np.r_[0:left,end:225]]),
            root_orientation_error_rad=angular(x,target,[0]),joins=[seam(x,fk,t) for t in (left,end)],
            raw_joins=[seam(raw,fk_24_np(raw),t) for t in (left,end)],dynamics=dynamics(x,fk),source_dynamics=dynamics(target,gt),
            immutable_bound=immutable_jerk_bound(fk,cfg["fixed_jerk_limit"],edit=(left,end)) if b["control"]=="correct" else None,
            contact_truth="UNKNOWN",source_proxy_support_frames=int(z["proxy_static"].sum()),physical=ref(folder/"physical.json"))
        editable=~tr["hard"][:,4:].all(-1);foot=fk[:,cfg["proxy"]["foot_joints"]]
        static=z["proxy_static"]&editable[:,None];segments=z["proxy_segment"]
        velocity_mask=z["proxy_static"][1:]&z["proxy_static"][:-1]&(segments[1:]==segments[:-1])&(editable[1:]|editable[:-1])[:,None]
        row["source_proxy_readouts"]=dict(support_elements=int(static.sum()),velocity_elements=int(velocity_mask.sum()),
            anchor_rmse_m=float(np.sqrt(np.mean((foot-z["proxy_anchor"])[static]**2))) if static.any() else None,
            velocity_rms_mps=float(np.sqrt(np.mean((np.diff(foot,axis=0)*30)[velocity_mask]**2))) if velocity_mask.any() else None,
            frozen_source_floor=float(z["proxy_floor"]))
        qrows=read(root/"rollout/result.json")["q_scores"][b["arm"]]["scores"]
        qr=next(q for q in qrows if q["case"]==b["case"] and q["variant"]==b["variant"] and q["t"]==19)
        eq=qr["unknown"] if b["control"]=="correct" else qr["controls"][b["control"]]
        actual=read(checked(b["sampler"]))["records"][-1]["unknown"]
        row.update(q_t19_unknown=eq,actual_t19_unknown=actual,actual_over_q=actual/max(eq,1e-30))
        if replay:
            oldrow=read(Path(replay)/"result.json")["records"][index]
            assert row["physical"]["sha256"]==oldrow["physical"]["sha256"]
            row["physical"]=oldrow["physical"]
        records.append(row);saved[str(index)]=b
        if replay:
            previous=Path(replay)/str(index)
            assert np.array_equal(x,np.load(previous/"native.npy")) and np.array_equal(fk,np.load(previous/"fk.npy"))
            assert physical==read(previous/"physical.json")
    result=dict(records=records,optimizer_steps=0,operational_qualified=False,whole_song_generations=0)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,saved=saved))
    print("R21_NATIVE",sum(r["native_ok"] for r in records),"/",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

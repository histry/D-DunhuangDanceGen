"""Fixed Native, raw balanced errors, bilateral responses and disjoint temporal domains."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.masked_denoiser.metrics import decode
from dunhuang.research.global_clock_r17.evaluate import score
from dunhuang.research.continuous_r16.evaluate import seam,angular
from dunhuang.research.task_denoising_r18.evaluate import dynamics
from dunhuang.research.objective_r19.readouts import balanced
from dunhuang.research.boundary_state_r14.bounds import immutable_jerk_bound

def run(cfg,root,out,part,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    from .common import parent_config,older_config,old_population
    pc=parent_config(cfg);pre,fit,ev=old_population(older_config(pc));native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from contracts.physical_quality import PhysicalQualityLimits
    scaler=np.load(checked(pre["assets"]["scaler"]));by={r["case"]:r for r in fit}
    records=[];saved={}
    for index,b in enumerate(read(root/(part+"_rollout")/"result.json")["bindings"]):
        z=np.load(checked(by[b["case"]]["arrays"]));tr=np.load(checked(b["trace"]))
        x=decode(tr["sample"],scaler,z["native"]);raw=decode(tr["raw_final"],scaler,z["native"])
        fk,physical=audit_native(x);gt=fk_24_np(z["native"]);rawfk=fk_24_np(raw)
        folder=out/str(index);folder.mkdir();np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"physical.json",physical)
        # Restore the correct mask for common-middle comparisons, including removed-side controls.
        from dunhuang.research.global_clock_r17.contract import conditions
        _,original,_=conditions(z["encoded"],"inbetween")
        losses,counts=balanced(tr["raw_final"],z["encoded"],original)
        jerk=np.linalg.norm(np.diff(fk,n=3,axis=0)*30**3,axis=-1);starts=np.arange(len(jerk));masks={}
        masks["fixed"]=sum(((starts<c)&(starts+3>=c)).astype(int) for c in (8,217))>0
        masks["fusion"]=(sum(((starts<c)&(starts+3>=c)).astype(int) for c in (75,150))>0)&~masks["fixed"]
        masks["phase"]=((starts<112)&(starts+3>=112))&~masks["fixed"]&~masks["fusion"]
        masks["interior"]=~(masks["fixed"]|masks["fusion"]|masks["phase"])
        domains={n:dict(templates=int(m.sum()),max=float(jerk[m].max()),p95=float(np.percentile(jerk[m],95))) for n,m in masks.items()}
        result=dict(case=b["case"],role=b["role"],arm=b["arm"],layout=b["layout"],seed=b["seed"],control=b["control"],
            native_ok=physical["absolute"]["ok"],**score(x,z["native"],fk,gt,np.r_[55:58,167:170]),
            raw_losses=losses,counts=counts,temporal_domains=domains,
            root_orientation_error_rad=angular(x,z["native"],[0]),
            boundary_root_velocity_error_mps=[float(np.linalg.norm((x[t,4:7]-x[t-1,4:7])-(z["native"][t,4:7]-z["native"][t-1,4:7]))*30) for t in (8,217)],
            raw_edge_pose_error_rad=angular(raw[np.r_[0:8,217:225]],z["native"][np.r_[0:8,217:225]]),
            dynamics=dynamics(x,fk),source_dynamics=dynamics(z["native"],gt),
            joins=[seam(x,fk,t) for t in (8,75,112,150,217)],raw_joins=[seam(raw,rawfk,t) for t in (8,75,112,150,217)],
            immutable_bound=immutable_jerk_bound(fk,PhysicalQualityLimits.from_environment().joint_jerk_mps3_max,edit=(8,217)) if b["control"] in ("correct","left_wrong","right_wrong") else None,
            contact_truth="UNKNOWN")
        records.append(result);saved[folder.name]=b
        if replay:
            other=Path(replay)/folder.name
            for f in folder.iterdir():
                if f.suffix==".npy":assert np.array_equal(np.load(f),np.load(other/f.name))
                else:assert read(f)==read(other/f.name)
    result=dict(records=records,optimizer_steps=0,part=part,whole_song_generations=0,operational_qualified=False)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,part=part,saved=saved))
    print("R20_NATIVE",part,sum(r["native_ok"] for r in records),"/",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--part",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.part,a.replay_from)

"""Unchanged Native plus local/raw boundaries, paired geometry, amplitude and source phase."""
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
from dunhuang.research.boundary_state_r14.bounds import immutable_jerk_bound
from .common import population

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pre,rows=population(cfg);native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from contracts.physical_quality import PhysicalQualityLimits
    scaler=np.load(checked(pre["assets"]["scaler"]));by={r["case"]:r for r in rows};records=[];saved={}
    for i,b in enumerate(read(root/"rollout/result.json")["bindings"]):
        z=np.load(checked(by[b["case"]]["arrays"]));tr=np.load(checked(b["trace"]))
        x=decode(tr["sample"],scaler,z["native"]);raw=decode(tr["raw_final"],scaler,z["native"])
        fk,physical=audit_native(x);gtfk=fk_24_np(z["native"]);rawfk=fk_24_np(raw)
        folder=out/str(i);folder.mkdir();np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"physical.json",physical)
        k=tr["known"];u=~k[:,4:];root_known=k[:,4:7]
        result=dict(case=b["case"],arm=b["arm"],control=b["control"],native_ok=physical["absolute"]["ok"],
            **score(x,z["native"],fk,gtfk,np.r_[55:58,167:170]),
            raw_unknown_mse_147=float(np.mean((tr["raw_final"][:,4:]-z["encoded"][:,4:])[u]**2)),
            raw_observed_root_rmse_m=float(np.sqrt(np.mean(((tr["raw_final"][:,4:7]-tr["condition"][:,4:7])/scaler["scale"][4:7])[root_known]**2))),
            dynamics=dynamics(x,fk),source_dynamics=dynamics(z["native"],gtfk),
            joins=[seam(x,fk,t) for t in (8,75,112,150,217)],
            raw_joins=[seam(raw,rawfk,t) for t in (8,75,112,150,217)],
            phase_pose_errors_rad=[angular(x[a:b],z["native"][a:b],list(range(12,20))) for a,b in ((0,112),(112,225))],
            immutable_bound=immutable_jerk_bound(fk,PhysicalQualityLimits.from_environment().joint_jerk_mps3_max,edit=(8,217)),
            contact_truth="UNKNOWN")
        records.append(result);saved[folder.name]=b
        if replay:
            other=Path(replay)/folder.name
            for f in folder.iterdir():
                if f.suffix==".npy":assert np.array_equal(np.load(f),np.load(other/f.name))
                else:assert read(f)==read(other/f.name)
    result=dict(records=records,optimizer_steps=0,whole_song_generations=0,
        human_reviewed=False,semantic_event_truth="UNKNOWN",operational_qualified=False)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,saved=saved))
    print("R19_NATIVE",sum(r["native_ok"] for r in records),"/",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

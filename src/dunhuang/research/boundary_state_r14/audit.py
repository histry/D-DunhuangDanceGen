"""Frozen R13 50-step responsibility and immutable-template audit; no generation."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write,sha
from dunhuang.data.sources import ref
from dunhuang.contracts import matrices
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.state_bridge.metrics import seam
from dunhuang.research.masked_denoiser.metrics import decode
from dunhuang.research.bridge_inpainting.study import audit_native
from .common import population
from .bounds import immutable_jerk_bound

def boundary_error(x,old,side):
    if side=="left":t=59;v=x[59,4:7]-x[58,4:7];a=x[59,4:7]-2*x[58,4:7]+x[57,4:7];g=old[59,4:7]-old[58,4:7];ga=old[59,4:7]-2*old[58,4:7]+old[57,4:7];lo,hi=58,59
    else:t=90;v=x[91,4:7]-x[90,4:7];a=x[92,4:7]-2*x[91,4:7]+x[90,4:7];g=old[91,4:7]-old[90,4:7];ga=old[92,4:7]-2*old[91,4:7]+old[90,4:7];lo,hi=90,91
    P=matrices(x[[lo,hi],7:13]);G=matrices(old[[lo,hi],7:13])
    w=Rotation.from_matrix(P[1]@P[0].T).as_rotvec()*30;gw=Rotation.from_matrix(G[1]@G[0].T).as_rotvec()*30
    return dict(root_position_error_m=float(np.linalg.norm(x[t,4:7]-old[t,4:7])),velocity_error_mps=float(np.linalg.norm(v-g)*30),
        acceleration_error_mps2=float(np.linalg.norm(a-ga)*900),root_angular_velocity_error_rps=float(np.linalg.norm(w-gw)))

def run(cfg,out,replay=None):
    out=fresh(out);assets,_,_,_,monitors=population(cfg);native_environment(assets)
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import OFFSETS,PARENTS
    from contracts.physical_quality import PhysicalQualityLimits
    limits=PhysicalQualityLimits.from_environment();scaler=np.load(checked(assets["scaler"]));data=Data(scaler)
    np.savez(out/"skeleton.npz",offsets=OFFSETS,parents=PARENTS)
    bounds=[];records=[];controls=[];arrays={}
    original=[]
    for i,b in enumerate(monitors):
        old=data.native(b["row"]);fk,physical=audit_native(old);original.append(old)
        bound=immutable_jerk_bound(fk,limits.joint_jerk_mps3_max,cfg["edit"])
        assert abs(bound["native_global_max"]-physical["audit"]["joint_jerk_mps3_max"])<1e-3
        bounds.append(dict(case=i,row=b["row"],original_native_ok=physical["absolute"]["ok"],**bound))
        arrays["original_"+str(i)]=old;arrays["fk_"+str(i)]=fk
    g=read(Path(cfg["r13_root"])/"main_rollout/result.json")
    for binding in g["bindings"]:
        if binding["driver"]!="correct":continue
        i=binding["case"];old=original[i];z=np.load(checked(binding["trace"]))
        for step,p in enumerate(z["raw"]):
            x=decode(p,scaler,old);hard=old.copy();hard[60:90]=x[60:90];fk=fk_24_np(hard)
            # These are frozen-state readouts except the final true rollout.
            records.append(dict(case=i,arm=binding["arm"],step=step,t=int(list(reversed(np.linspace(-1,999,51).astype(int).tolist()))[step]),
                raw_left=boundary_error(x,old,"left"),raw_right=boundary_error(x,old,"right"),
                raw_relative_root_error_m=float(np.linalg.norm((x[90,4:7]-x[59,4:7])-(old[90,4:7]-old[59,4:7]))),
                graft_seams=[seam(hard,fk,t,2e-5) for t in (60,90)]))
            if step==49:
                expected=np.load(Path(cfg["r13_root"])/"main_native"/(str(i)+"_"+binding["arm"]+"_correct")/"native.npy")
                assert np.array_equal(hard,expected)
        names=("correct","removed","removed_left","removed_right","wrong_left","wrong_right")
        for j,timestep in enumerate(z["t"]):
            for k,name in enumerate(names):
                x=decode(z["prediction"][j,k],scaler,old)
                controls.append(dict(case=i,arm=binding["arm"],t=int(timestep),control=name,
                    left=boundary_error(x,old,"left"),right=boundary_error(x,old,"right")))
    np.savez_compressed(out/"originals.npz",**arrays)
    cached=[b for b in read(Path(cfg["r13_root"])/"main/result.json")["self_caches"] if b["update"]==250][:cfg["conflict_cases"]]
    np.savez_compressed(out/"train_parity.npz",**{str(i):fk_24_np(data.native(b["row"])) for i,b in enumerate(cached)})
    result=dict(records=records,controls=controls,necessary_bounds=bounds,
        necessary_excluded_cases=sum(b["necessary_status"]!="UNKNOWN" for b in bounds),
        unknown_cases=sum(b["necessary_status"]=="UNKNOWN" for b in bounds),optimizer_steps=0,new_neural_generations=0,
        all_four_cases_retained=True,decomposition_is_causal=False,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        frozen_r13_generation=ref(Path(cfg["r13_root"])/"main_rollout/result.json"),skeleton=ref(out/"skeleton.npz"),
        native_jerk_bound_dtype="Native float32 third finite difference, same 30Hz/global maximum",other_metrics_necessary_status="UNKNOWN"))
    if replay:
        assert result==read(Path(replay)/"result.json")
        from dunhuang.research.source_tasks.common import same_arrays
        same_arrays(out/"originals.npz",Path(replay)/"originals.npz");same_arrays(out/"skeleton.npz",Path(replay)/"skeleton.npz");same_arrays(out/"train_parity.npz",Path(replay)/"train_parity.npz")
    print("R14_FROZEN_AUDIT",len(records),result["necessary_excluded_cases"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)

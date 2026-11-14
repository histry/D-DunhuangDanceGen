"""Native evaluation of single-state capacity probes, never output Root repair."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import write,checked
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import Data
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.masked_denoiser.metrics import physical
from .common import population

def run(cfg,root,out,replay=None):
    out=fresh(out);forbid_optimizers();root=Path(root);base=read(cfg["base_config"])
    assets,_,_,_,_=population(base);native_environment(assets)
    scaler=np.load(checked(assets["scaler"]));data=Data(scaler);records=[];bindings=read(root/"capacity/result.json")["bindings"]
    for b in bindings:
        old=data.native(b["row"]);z=np.load(b["probe"])
        for name in z.files:
            native,fk,audit,m=physical(z[name],old,scaler);folder=out/(str(b["case"])+"_"+name);folder.mkdir()
            np.save(folder/"native.npy",native);np.save(folder/"fk.npy",fk);write(folder/"physical.json",audit)
            arm,t=name.rsplit("_",1)
            records.append(dict(case=b["case"],arm=arm,t=int(t),**m))
            if replay:
                other=Path(replay)/folder.name
                assert np.array_equal(native,np.load(other/"native.npy")) and np.array_equal(fk,np.load(other/"fk.npy"))
                assert audit==read(other/"physical.json")
    baseline={(r["case"],r["t"]):r for r in records if r["arm"]=="baseline"}
    quality={}
    for arm in cfg["reference_arms"]:
        rr=[r for r in records if r["arm"]==arm];checks=[]
        for r in rr:
            b=baseline[r["case"],r["t"]]
            # No pooling away a bad right/left seam. Source pose is evaluated against real target.
            seams=all(s[k]<=v[k]*(1+cfg["seam_relative_tolerance"])+1e-8
                for s,v in zip(r["joins"],b["joins"]) for k in ("root_step_m","root_turn_rad","foot_step_max_m","peak_jerk_mps3"))
            checks.append(dict(case=r["case"],t=r["t"],both_seams_nonregression=seams,
                source_pose_protected=r["pose_error_rad"]<=b["pose_error_rad"]+cfg["pose_tolerance_rad"],
                root_improved=r["root_rmse_m"]<b["root_rmse_m"],native_ok=r["native_ok"]))
        quality[arm]=checks
    gains={}
    correct=[r for r in records if r["arm"]=="state_A2"]
    for control in ("raw_A0","shuffled_A2"):
        other={(r["case"],r["t"]):r for r in records if r["arm"]==control}
        groups=[]
        for case in range(cfg["cases"]):
            a=np.mean([r["root_rmse_m"] for r in correct if r["case"]==case])
            b=np.mean([r["root_rmse_m"] for r in other.values() if r["case"]==case])
            groups.append(float(1-a/b))
        gains[control]=dict(group_gains=groups,wins=sum(g>0 for g in groups),mean_group_gain=float(np.mean(groups)))
    deriv=read(root/"capacity/result.json")["derivatives"]
    numeric_ok=all(r["derivative_relative_error"]<=.05 or abs(r["derivative_numeric"]-r["derivative_analytic"])<=1e-4 for r in deriv)
    all_quality=all(all(r[k] for k in ("both_seams_nonregression","source_pose_protected","root_improved")) for r in quality["state_A2"])
    gate=bool(numeric_ok and all_quality and all(g["wins"]>=cfg["minimum_group_wins"] and g["mean_group_gain"]>=cfg["minimum_gain"] for g in gains.values()))
    result=dict(records=records,quality=quality,root_gains=gains,numeric_gradient_passed=numeric_ok,
        single_state_capacity_passed=gate,new_population_training_admitted=False,
        next_gate="If capacity passes, register full self-policy guided sampling before any neural training",
        optimizer_steps=0,operational_qualified=False,new_whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,scope="Native readouts of raw x0 at two frozen policy states"))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R14_CAPACITY_GATE",gate,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

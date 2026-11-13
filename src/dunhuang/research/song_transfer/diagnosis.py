"""Saved full-song jerk localization; association is not causal attribution."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh

def stats(x):
    v=np.asarray(x,float).ravel()
    return dict(count=len(v),mean=float(v.mean()),p95=float(np.percentile(v,95)),maximum=float(v.max())) if len(v) else dict(count=0,mean=None,p95=None,maximum=None)

def run(generation,out,replay_from=None):
    out=fresh(out);protocol=read(Path(generation)/"protocol.json");rows=[]
    for b in protocol["bindings"]:
        x=np.load(checked(b["fk"]));q=read(checked(b["request"]));physical=read(checked(b["physical"]))
        j=np.diff(x,n=3,axis=0)*30**3;n=np.linalg.norm(j,axis=-1)
        audit=physical["audit"];assert abs(float(np.percentile(n.astype(float),95))-audit["joint_jerk_mps3_p95"])<1e-6
        centers=np.arange(len(n))+1.5;boundaries=75*np.arange(1,q["windows"])
        band=(abs(centers[:,None]-boundaries[None,:])<=3).any(1)
        high=n>physical["absolute"]["limits"]["joint_jerk_mps3_max"]
        value=x.astype(float);root=np.diff(value[:,:1],n=3,axis=0)*30**3
        relative=np.diff(value-value[:,:1],n=3,axis=0)*30**3
        residual=float(abs(np.diff(value,n=3,axis=0)*30**3-root-relative).max())
        assert residual<1e-8
        rows.append(dict(case=b["case"],arm=b["arm"],frames=len(x),derivative_stencil="forward third difference; center k+1.5",
            boundary_band_frames=3,boundary_samples=int(band.sum()),all_samples=len(band),
            world_jerk_mps3=stats(n),boundary_world_jerk_mps3=stats(n[band]),interior_world_jerk_mps3=stats(n[~band]),
            high_joint_frame_threshold_mps3=physical["absolute"]["limits"]["joint_jerk_mps3_max"],
            high_joint_frames=int(high.sum()),high_joint_frames_at_boundaries=int(high[band].sum()),high_joint_frames_interior=int(high[~band].sum()),
            root_translation_jerk_mps3=stats(np.linalg.norm(root,axis=-1)),root_relative_jerk_mps3=stats(np.linalg.norm(relative,axis=-1)),
            vector_decomposition_max_abs=residual,causal_attribution=False,observed_contact_truth=False))
    result=dict(records=rows,optimizer_calls=0,new_generation_calls=0,thresholds_changed=False,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),generation=ref(Path(generation)/"protocol.json"),
        method="all saved full songs, no population selection; fixed +/-3-frame bands at computational window starts; vector translation decomposition is not causal"))
    if replay_from:
        if result!=read(Path(replay_from)/"result.json"):raise ValueError("Saved-output diagnosis replay differs")
        write(out/"replay.json",dict(status="FRESH_SAVED_OUTPUT_DIAGNOSIS_REPLAY",optimizer_calls=0,new_generation_calls=0))
    print("R7_SAVED_DIAGNOSIS",len(rows),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--generation",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(a.generation,a.out,a.replay_from)

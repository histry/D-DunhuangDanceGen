"""Common-span, actual external graft, source-response and unchanged Native reports."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write
from dunhuang.contracts import matrices
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.masked_denoiser.metrics import decode
from .gate import qualification

FEET=[7,8,10,11]

def angular(x,y,joints=None):
    R=matrices(x[:,7:].reshape(-1,24,6));G=matrices(y[:,7:].reshape(-1,24,6))
    if joints is not None:R=R[:,joints];G=G[:,joints]
    return float(Rotation.from_matrix((G.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().mean())

def seam(x,fk,t):
    jerk=np.linalg.norm(np.diff(fk,n=3,axis=0)*30.**3,axis=-1)
    starts=np.arange(len(jerk));cross=(starts<t)&(starts+3>=t)
    R=matrices(x[[t-1,t],7:13])
    return dict(frame=t,root_step_m=float(np.linalg.norm(x[t,4:7]-x[t-1,4:7])),
        root_turn_rad=float(Rotation.from_matrix(R[0].T@R[1]).magnitude()),
        foot_step_max_m=float(np.linalg.norm(fk[t,FEET]-fk[t-1,FEET],axis=-1).max()),
        crossing_jerk_max_mps3=float(jerk[cross].max()))

def run(cfg,preflight,samples,out,replay=None):
    out=fresh(out);forbid_optimizers();pre=read(Path(preflight)/"protocol.json");native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from contracts.physical_quality import PhysicalQualityLimits
    limit=PhysicalQualityLimits.from_environment().joint_jerk_mps3_max
    scaler=np.load(checked(pre["assets"]["scaler"]));records=[]
    for b in read(Path(samples)/"result.json")["bindings"]:
        i=b["case"];arm=b["arm"];z=np.load(checked(pre["pairs"][i]["arrays"]))
        target=z["native"][:150];trace=np.load(checked(b["trace"]));raw=decode(trace["sample"][0],scaler,target)
        if arm=="C0":
            x=target.copy();x[60:90]=raw[60:90]
        else:x=raw
        fk,audit=audit_native(x);halo=z["halo"].copy();halo[8:158]=x
        hfk,haudit=audit_native(halo);gtfk=fk_24_np(target)
        folder=out/(str(i)+"_"+arm);folder.mkdir()
        for name,array in (("native.npy",x),("fk.npy",fk),("halo.npy",halo),("halo_fk.npy",hfk)):np.save(folder/name,array)
        write(folder/"physical.json",audit);write(folder/"halo_physical.json",haudit)
        jerk=np.linalg.norm(np.diff(fk,n=3,axis=0)*30.**3,axis=-1)
        keys=np.unique(np.concatenate([np.arange(t-1,t+2) for t in z["keyframes"]]))
        edge=np.r_[0:8,142:150]
        activity=float(np.linalg.norm(np.diff(fk[8:142],axis=0)*30,axis=-1).mean())
        gtactivity=float(np.linalg.norm(np.diff(gtfk[8:142],axis=0)*30,axis=-1).mean())
        r=dict(case=i,arm=arm,global_jerk_max_mps3=float(jerk.max()),center_jerk_max_mps3=float(jerk[44:105].max()),
            event_join=seam(x,fk,cfg["event_boundaries"][i]),legacy_joins=[seam(x,fk,t) for t in (60,90)],
            external_joins=[seam(halo,hfk,t) for t in (8,158)],edge_root_rmse_m=float(np.sqrt(np.mean((x[edge,4:7]-target[edge,4:7])**2))),
            event_pose_error_rad=angular(x[keys],target[keys],cfg["source_joints"]),
            common_generated_span_pose_rad=angular(x[60:90],target[60:90]),
            activity_ratio=activity/max(gtactivity,1e-8),native_ok=audit["absolute"]["ok"],halo_native_ok=haudit["absolute"]["ok"],
            contact_truth="UNKNOWN",parent_outside30_preserved=arm=="C0",source_pose_paste=arm=="C0")
        records.append(r)
        if replay:
            other=Path(replay)/folder.name
            for f in ("native.npy","fk.npy","halo.npy","halo_fk.npy"):assert np.array_equal(np.load(folder/f),np.load(other/f))
            for f in ("physical.json","halo_physical.json"):assert read(folder/f)==read(other/f)
    gate=qualification(records,cfg,limit)
    result=dict(records=records,qualification=gate,optimizer_steps=0,operational_qualified=False,new_whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,native_jerk_limit=limit,
        metrics_scope="C0 copies real parent outside30; C1 generates all150. Source errors compared on common generated30 separately. All new outer joins retained."))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R16_FROZEN_RESPONSE_GATE",gate["cross_window_training_admitted"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--preflight",required=True);p.add_argument("--samples",required=True)
    p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.preflight,a.samples,a.out,a.replay_from)

"""Source identity, stages, dynamics and music response, distinct from Native."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from .reference import build
def angles(a,b):
    return Rotation.from_matrix((a.swapaxes(-1,-2)@b).reshape(-1,3,3)).magnitude().reshape(a.shape[:-2])
def measure(native,plan,cfg):
    reference,known,weights,phases=build(plan,cfg);T=len(native)
    a=matrices(native[:,7:].reshape(T,24,6));b=matrices(reference[:,7:].reshape(T,24,6))
    errors=angles(a,b);points=set()
    for p in phases:
        t=min(T-1,max(0,round(p["music_frame"])))
        points.update((t,j) for j in p["joints"] if 1<=j<22)
    av=angles(a[:-1,1:22],a[1:,1:22])*30;bv=angles(b[:-1,1:22],b[1:,1:22])*30
    turns=[]
    for e in plan["events"]:
        l,h=e["music_interval"];pa=a[l,0].T@a[l:h,0];pb=b[l,0].T@b[l:h,0];turns.extend(angles(pa,pb).tolist())
    from motion.geometry import fk_24_np
    pa=fk_24_np(native);pb=fk_24_np(reference)
    ca=(pa-pa[:,:1])@a[:,0];cb=(pb-pb[:,:1])@b[:,0]
    wrist_elbow=float(np.linalg.norm(ca[:,(18,19,20,21)]-cb[:,(18,19,20,21)],axis=-1).mean())
    hold_a=av[:,11:19].mean(-1)<=.2;hold_b=bv[:,11:19].mean(-1)<=.2
    return dict(wrist_elbow_root_frame_m=wrist_elbow,wrist_rotation_rad=float(errors[:,20:22].mean()),trunk_rotation_rad=float(errors[:,(3,6,9,12)].mean()),
        generated_hold_seconds=float(hold_a.sum()/30),source_hold_seconds=float(hold_b.sum()/30),
        hold_proxy_coverage=float((hold_a&hold_b).sum()/hold_b.sum()) if hold_b.sum()>=6 else None,
        mean_source_rad=float(errors[:,1:22].mean()),phase_pose_rad=float(np.mean([errors[t,j] for t,j in sorted(points)])),
        phase_joint_points=len(points),root_relative_turn_rad=float(np.mean(turns)),activity_ratio=float(av.mean()/max(bv.mean(),1e-8)),
        generated_local_speed_radps=float(av.mean()),source_local_speed_radps=float(bv.mean()),reference_frames=T,observed_joints=list(range(22)),semantic_quality="UNKNOWN")
def response(a,b):
    return float(angles(matrices(a[:,7:139].reshape(-1,22,6))[:,1:],matrices(b[:,7:139].reshape(-1,22,6))[:,1:]).mean())
def decide(records,cfg):
    full=[r for r in records if r["arm"]=="S3"];passed=[]
    for r in full:
        m=r["source"];base=next(x for x in records if x["pair"]==r["pair"] and x["music"]==r["music"] and x["arm"]=="baseline")
        own=base["sources"][r["source_label"]]["mean_source_rad"]-m["mean_source_rad"]
        other=base["sources"]["B" if r["source_label"]=="A" else "A"]["mean_source_rad"]-r["other_source"]["mean_source_rad"]
        lo,hi=cfg["activity_ratio"]
        ok=m["mean_source_rad"]<=cfg["source_mean_rad"] and m["phase_pose_rad"]<=cfg["key_pose_mean_rad"] and m["root_relative_turn_rad"]<=cfg["root_turn_mean_rad"] and lo<=m["activity_ratio"]<=hi and m["wrist_rotation_rad"]<=cfg["wrist_rotation_rad"] and m["trunk_rotation_rad"]<=cfg["trunk_rotation_rad"] and m["wrist_elbow_root_frame_m"]<=cfg["wrist_elbow_root_frame_m"] and (m["hold_proxy_coverage"] is None or m["hold_proxy_coverage"]>=cfg["hold_proxy_coverage"]) and own>=cfg["own_gain_rad"] and own-other>=cfg["relative_gain_rad"]
        passed.append(dict(pair=r["pair"],music=r["music"],source_label=r["source_label"],own_gain_rad=own,directional_gain_rad=own-other,source_ok=ok,native_ok=r["native_ok"]))
    return dict(source_passes=sum(x["source_ok"] for x in passed),native_passes=sum(x["native_ok"] for x in passed),denominator=len(passed),records=passed)

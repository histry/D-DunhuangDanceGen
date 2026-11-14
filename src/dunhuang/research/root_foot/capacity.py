import argparse,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,pinned,sha
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.bridge_inpainting.study import audit_native

def solve(feet,support,cfg):
    T=len(feet);D=np.diff(np.eye(T),axis=0);D3=np.diff(np.eye(T),n=3,axis=0)
    active=support[:-1]&support[1:];weights=active.sum(1)
    velocity=np.diff(feet[...,(0,2)].astype(float),axis=0)*30
    mean=np.divide((velocity*active[...,None]).sum(1),weights[:,None],out=np.zeros((T-1,2)),where=weights[:,None]>0)
    irreducible=((velocity-mean[:,None])**2).sum(-1)
    H=cfg["root_l2"]*np.eye(T)+cfg["root_d3"]*(D3.T@D3)+900*D.T@(weights[:,None]*D)
    rhs=-30*D.T@(velocity*active[...,None]).sum(1)
    free=np.ones(T,bool);free[:3]=False;free[-3:]=False
    x=np.zeros((T,2));x[free]=np.linalg.solve(H[np.ix_(free,free)],rhs[free])
    amp=np.linalg.norm(x,axis=-1).max();speed=np.linalg.norm(np.diff(x,axis=0)*30,axis=-1).max()
    fraction=min(1.,cfg["radius_m"]/max(amp,1e-12),cfg["delta_speed_cap_mps"]/max(speed,1e-12));x*=fraction
    evidence=dict(least_squares_unrestricted_root_max_m=float(amp),least_squares_delta_speed_max_mps=float(speed),trust_scale=float(fraction),
        fixed_support_edges=int(active.sum()),multiple_support_edges=int((weights>=2).sum()),
        irreducible_horizontal_velocity_squared=float(irreducible[active].sum()),unrestricted_per_edge_compensation_mps=mean.tolist(),
        fixed_support_truth=False,capacity_scope="finite trust-scaled linear least-squares proposal, not a global impossibility proof")
    return x,evidence,active

def frozen_support(native,points):
    from motion.physics import derive_contacts_np
    from motion.config import MotionGenerationConfig
    from motion_geometry.physical import classify_support_states_np,median_filter_bool_np,STATIC_SUPPORT
    cfg=MotionGenerationConfig();contacts,_,_,_=derive_contacts_np(native,cfg,precomputed_joints=points)
    feet=points[:,[7,8,10,11]];floor=float(np.percentile(feet[...,1],5))
    height=median_filter_bool_np(feet[...,1]<=floor+.055,3)
    states=classify_support_states_np(points,contacts>.5,fps=30.,sliding_support_eligible=None,height_support=height,support_policy="final_fail_closed")
    return states==STATIC_SUPPORT

def root_stats(x):
    v=np.linalg.norm(np.diff(x[:,[4,6]],axis=0)*30,axis=-1)
    return dict(speed_p95=float(np.percentile(v,95)),path_m=float(np.linalg.norm(np.diff(x[:,[4,6]],axis=0),axis=-1).sum()),
        net_m=float(np.linalg.norm(x[-1,[4,6]]-x[0,[4,6]])))

def run(cfg,out,replay=None):
    out=fresh(out);root=Path(cfg["parent"]);seal=read("reports/r9_summary.json");assert sha(root/"source/protocol.json")==seal["evidence"]["r9/source/protocol.json"]
    protocol=read(root/"source/protocol.json");rows=read(root/"source/result.json")["records"];assets=read("inputs/clock_assets_v2.json")
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from motion.geometry import fk_24_np
    baselines={(b["pair"],b["source"]):b for b in protocol["bindings"] if b["arm"]=="T0"};records=[];bindings=[]
    for b in protocol["bindings"]:
        if b["arm"] not in cfg["arms"]:continue
        old=np.load(checked(b["native"]));points=fk_24_np(old)
        base=baselines[(b["pair"],b["source"])];t0=np.load(checked(base["native"]));mask=frozen_support(t0,fk_24_np(t0))
        baseline_audit=next(r for r in rows if r["pair"]==b["pair"] and r["source"]==b["source"] and r["arm"]=="T0")["audit"]
        assert abs(mask.mean()-baseline_audit["static_support_ratio"])<1e-8
        before=next(r for r in rows if r["pair"]==b["pair"] and r["source"]==b["source"] and r["arm"]==b["arm"])["audit"]
        for radius in cfg["radii_m"]:
            folder=out/(str(b["pair"])+"_"+b["source"]+"_"+b["arm"]+"_"+str(radius));folder.mkdir()
            delta,evidence,active=solve(points[:,[7,8,10,11]],mask,dict(cfg,radius_m=radius))
            x=old.copy();x[:,[4,6]]+=delta.astype(np.float32)
            assert np.array_equal(x[:,7:],old[:,7:]) and np.array_equal(x[:,:4],old[:,:4]) and np.array_equal(x[:,5],old[:,5])
            assert np.array_equal(x[:3],old[:3]) and np.array_equal(x[-3:],old[-3:])
            fk,p=audit_native(x);speed=np.linalg.norm(np.diff(fk[:,[7,8,10,11]][...,(0,2)],axis=0)*30,axis=-1)
            oldspeed=np.linalg.norm(np.diff(points[:,[7,8,10,11]][...,(0,2)],axis=0)*30,axis=-1)
            fixed_before=float(np.percentile(oldspeed[active],95)) if active.any() else None;fixed_after=float(np.percentile(speed[active],95)) if active.any() else None
            guards=dict(fixed_support_improves=bool(active.any() and fixed_after<=.9*fixed_before),root_speed_nonregression=root_stats(x)["speed_p95"]<=root_stats(old)["speed_p95"]+1e-6,
                jerk_nonregression=p["audit"]["joint_jerk_mps3_p95"]<=before["joint_jerk_mps3_p95"]+1e-6,
                penetration_unchanged=abs(p["audit"]["foot_penetration_min_m"]-before["foot_penetration_min_m"])<1e-5)
            row=dict(pair=b["pair"],source=b["source"],arm=b["arm"],radius_m=radius,proposal=evidence,guards={k:bool(v) for k,v in guards.items()},
                fixed_support_p95_before=fixed_before,fixed_support_p95_after=fixed_after,root_before=root_stats(old),root_after=root_stats(x),
                audit=p["audit"],native_ok=p["absolute"]["ok"],research_pass=all(guards.values()),poses_and_contacts_bitwise_preserved=True)
            records.append(row);np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);np.save(folder/"support.npy",mask);write(folder/"physical.json",p)
            bindings.append(dict(name=folder.name,native=ref(folder/"native.npy"),fk=ref(folder/"fk.npy"),support=ref(folder/"support.npy"),original=b["native"],baseline=base["native"]))
            if replay:
                other=Path(replay)/folder.name
                for f in ("native.npy","fk.npy","support.npy"):assert np.array_equal(np.load(folder/f),np.load(other/f))
                assert p==read(other/"physical.json")
    result=dict(records=records,population=len(records),research_passes=sum(r["research_pass"] for r in records),native_passes=sum(r["native_ok"] for r in records),new_neural_generations=0,optimizer_updates=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,parent=ref(root/"source/protocol.json"),bindings=bindings))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R10_ROOT_CAPACITY",len(records),result["research_passes"],result["native_passes"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

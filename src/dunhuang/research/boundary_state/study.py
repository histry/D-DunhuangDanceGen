"""Finite R9-B0/B1 proposals; no new neural sampling or global feasibility claim."""
import argparse,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from dunhuang.io import checked,write,sha,pinned
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.bridge_inpainting.study import audit_native,source_error,region,local_plan
from .projection import project

def activity(x,a,b):
    r=matrices(x[a:b,7:].reshape(-1,24,6))[:,1:22]
    return float(Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).magnitude().mean()*30)

def stats(fk,a,b):
    j=np.linalg.norm(np.diff(fk,n=3,axis=0)*27000,axis=-1)[max(0,a-3):b]
    return dict(peak_jerk_mps3=float(j.max()),mean_joint_jerk_area_mps2=float(j.mean(axis=1).sum()/30))

def keys(plan,a,b):
    result={}
    for event in plan["events"]:
        warp=np.asarray(event["time_warp"])
        for phase in event["full_phases"]:
            t=round(float(np.interp(phase["frame"],warp[:,0],warp[:,1])))
            if a+3<=t<b-3:result.setdefault(t,set()).update(phase["observed_joints"])
    return {t:sorted(j) for t,j in result.items()}

def run(cfg,out,replay=None):
    out=fresh(out);root=Path(cfg["parent"]);sealed=read("reports/r8_summary.json")
    for f in ("bridge/protocol.json","bridge/result.json","boundary_audit/protocol.json","boundary_audit/candidates.json"):assert sha(root/f)==sealed["evidence"][f]
    original=read(root/"boundary_audit/candidates.json");bp=read(root/"bridge/protocol.json")
    assets=read(checked(read(root/"boundary_audit/protocol.json")["assets"]))
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    records=[];bindings=[];a,b=cfg["projection"]["edit"]
    for c in original:
        old=np.load(checked(c["native"]));prior_fk=np.load(checked(c["fk"]));plan=read(checked(c["event_plan"]))
        reference=old.copy()
        for lo,hi,value in local_plan(plan,a,b):reference[lo:hi,7:139]=value[:,7:139]
        ref_speed=activity(reference,a,b)
        for arm in cfg["candidates"]:
            binding=next(v for v in bp["bindings"] if v["name"]==c["name"] and v["arm"]==arm)
            raw=np.load(checked(binding["motion"]));raw_fk=np.load(checked(binding["fk"]));raw_source=source_error(raw,plan,a,b)
            raw_activity=abs(np.log(max(activity(raw,a,b),1e-8)/max(ref_speed,1e-8)))
            for domain in cfg["domains"]:
                for key_policy in cfg["key_policies"]:
                    name=c["name"]+"_"+arm+"_"+domain["name"]+"_"+key_policy;folder=out/name;folder.mkdir()
                    q=dict(cfg["projection"],**{k:v for k,v in domain.items() if k!="name"})
                    x,capacity=project(old,raw,q,keys(plan,a,b) if key_policy=="source_keys" else {})
                    row=dict(name=name,case=c["name"],candidate_arm=arm,domain=domain["name"],key_policy=key_policy,capacity=capacity,
                        candidate_source_rad=raw_source,candidate_activity_deviation=raw_activity,new_neural_generations=0,research_local_pass=False,formal_qualified=False)
                    if x is not None:
                        fk,global_native=audit_native(x);_,roi=audit_native(x[:150])
                        center=region(fk,69,81);joins=[region(fk,t-3,t+3) for t in (a,b)];oldjoins=[region(prior_fk,t-3,t+3) for t in (a,b)]
                        sr=source_error(x,plan,a,b);act=abs(np.log(max(activity(x,a,b),1e-8)/max(ref_speed,1e-8)))
                        guards=dict(boundary_state=all(v["root_position_residual_m"]<2e-6 and v["root_d2_residual_mps2"]<.01 and v["rotation_residual_rad"]<2e-5 and v["angular_cross_seam_d2_max_radps3"]<.5 for v in capacity["boundary_contract"]),
                            new_joins_nonregression=all(v["maximum"]<=w["maximum"]+1e-6 for v,w in zip(joins,oldjoins)),
                            source_nonregression=sr<=raw_source+1e-6,activity_nonregression=act<=raw_activity+1e-6,
                            center_retains_improvement=center["p95"]<=region(prior_fk,69,81)["p95"]*.9)
                        row.update(guards=guards,research_local_pass=all(guards.values()),formal_qualified=all(guards.values()) and roi["absolute"]["ok"] and global_native["absolute"]["ok"],
                            source_rad=sr,activity_deviation=act,center=center,joins=joins,old_joins=oldjoins,roi_native_ok=roi["absolute"]["ok"],global_native_ok=global_native["absolute"]["ok"],
                            candidate_jerk=stats(raw_fk,a,b),projected_jerk=stats(fk,a,b))
                        np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"roi_physical.json",roi);write(folder/"global_physical.json",global_native)
                        bindings.append(dict(name=name,native=ref(folder/"native.npy"),fk=ref(folder/"fk.npy"),original=c["native"],candidate=binding["motion"],config=q))
                        if replay:
                            other=Path(replay)/folder.name
                            for file in ("native.npy","fk.npy"):assert np.array_equal(np.load(folder/file),np.load(other/file))
                            for file in ("roi_physical.json","global_physical.json"):assert read(folder/file)==read(other/file)
                    write(folder/"capacity.json",capacity);records.append(row);print("R9_BOUNDARY",name,capacity["status"],row["research_local_pass"],flush=True)
    result=dict(records=records,research_passes=sum(r["research_local_pass"] for r in records),formal_passes=sum(r["formal_qualified"] for r in records),
        population=len(records),new_neural_generations=0,new_whole_song_generations=0,neural_weight_updates=0,
        capacity_scope="finite registered proposals; necessary-bound rejects certify only impossibility within that cap",operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,parent=ref(root/"bridge/protocol.json"),bindings=bindings))
    if replay:
        assert result==read(Path(replay)/"result.json");write(out/"replay.json",dict(result_equal=True,projected_arrays_equal=len(bindings)*2,neural_generations=0))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

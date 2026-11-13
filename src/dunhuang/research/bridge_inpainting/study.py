"""Fixed local repair controls; all outside-edit motion is protected byte for byte."""
import argparse,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices,sixd
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.research.primitive_reperformance.environment import worker_environment
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.primitive_reperformance.metrics import measure

def smooth(native,cfg):
    x=native.copy();a,b=cfg["edit"];lo=max(0,a-8);hi=min(len(x),b+8);r=Rotation.from_matrix(matrices(x[lo:hi,7:].reshape(-1,24,6)).reshape(-1,3,3))
    r0=Rotation.from_matrix(matrices(x[lo,7:].reshape(24,6)));origin=Rotation.from_matrix(np.tile(r0.as_matrix()[None],(hi-lo,1,1,1)).reshape(-1,3,3))
    tangent=(origin.inv()*r).as_rotvec().reshape(hi-lo,24,3)
    filtered=savgol_filter(tangent,cfg["smooth_window"],cfg["smooth_order"],axis=0)
    u=np.linspace(0,1,b-a);envelope=np.sin(np.pi*u)**4;envelope[:2]=0;envelope[-2:]=0
    local=tangent[a-lo:b-lo]+envelope[:,None,None]*(filtered[a-lo:b-lo]-tangent[a-lo:b-lo])
    out=r0.as_matrix()[None]@Rotation.from_rotvec(local.reshape(-1,3)).as_matrix().reshape(b-a,24,3,3)
    x[a:b,7:]=sixd(out).reshape(b-a,144).astype(np.float32)
    root=savgol_filter(native[lo:hi,4:7],cfg["smooth_window"],cfg["smooth_order"],axis=0)
    x[a:b,4:7]+=envelope[:,None]*(root[a-lo:b-lo]-native[a:b,4:7])
    return x

def audit_native(x):
    from motion.geometry import fk_24_np
    from motion.physics import audit_motion_np
    from contracts.physical_quality import evaluate_physical_audit
    p=fk_24_np(x);audit=audit_motion_np(x,precomputed_joints=p);return p,dict(audit=audit,absolute=evaluate_physical_audit(audit))
def region(p,lo,hi):
    n=np.linalg.norm(np.diff(p,n=3,axis=0)*30**3,axis=-1);centers=np.arange(len(n))+1.5;m=(centers>=lo)&(centers<=hi)
    return dict(p95=float(np.percentile(n[m].astype(float),95)),maximum=float(n[m].max()))
def local_plan(full,a,b):
    from dunhuang.generation.soft_reference import resample_native
    from dunhuang.planning.time_map import source_positions
    # Caller evaluates directly on source-derived arrays for a partial bridge ROI.
    result=[]
    for e in full["events"]:
        l,h=e["music_interval"];L=max(a,l);H=min(b,h)
        if H>L:result.append((L,H,resample_native(np.load(checked(e["reference"])),source_positions(e,np.arange(L,H)))))
    return result
def source_error(x,plan,a,b):
    from dunhuang.research.primitive_reperformance.metrics import angles
    errors=[]
    for l,h,refmotion in local_plan(plan,a,b):
        A=matrices(x[l:h,7:].reshape(-1,24,6))[:,1:22];B=matrices(refmotion[:,7:].reshape(-1,24,6))[:,1:22];errors.extend(angles(A,B).ravel().tolist())
    return float(np.mean(errors))

def run(cfg,rconfig,audit,out,replay_from=None):
    out=fresh(out);p=read(Path(audit)/"protocol.json");assets=read(checked(p["assets"]));sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    candidates=read(checked(p["candidates"]));records=[];bindings=[];reject=[]
    for candidate in candidates:
        name=candidate["name"];old=np.load(checked(candidate["native"]));old_fk=np.load(checked(candidate["fk"]));plan=read(checked(candidate["event_plan"]))
        a,b=cfg["edit"];protected=np.ones(len(old),bool);protected[a:b]=False;before_source=source_error(old,plan,a,b)
        for arm in cfg["arms"]:
            folder=out/(name+"_"+arm);noise=None
            if arm=="smooth":folder.mkdir();edited=smooth(old,cfg)
            elif not candidate["context_supported"]:
                reject.append(dict(name=name,arm=arm,status="REJECTED_ACTUAL_CONTEXT_OUTSIDE_NORMALIZER",unsupported_coordinates=candidate["unsupported_coordinates"]));continue
            else:
                folder.mkdir();request=dict(assets,audio=candidate["audio"],features=candidate["features"],frames=150,windows=1,mode="anchored",anchors=[],
                    seed=cfg["seed"]+cfg["cases"].index(name),overlap_policy="mean",research_bridge_v1=True,bridge_context=candidate["context"],
                    bridge_source_condition=arm=="event_inpaint",event_plan=candidate["event_plan"],reference_offset=0,edit_interval=cfg["edit"],
                    condition_group="S3_bridge",reperformance_config=rconfig["guide"],generation_stage="R8_BIDIRECTIONAL_LOCAL_INPAINT")
                request["output"]=str(folder.resolve());write(folder/"request.json",request)
                for python,module in ((assets["edge_python"],"dunhuang.research.bridge_inpainting.worker"),(assets["native_python"],"dunhuang.physical")):
                    with (folder/(module.split(".")[-1]+".log")).open("w") as log:
                        subprocess.run([python,"-m",module,str((folder/"request.json").resolve())],stdout=log,stderr=subprocess.STDOUT,check=True,env=worker_environment())
                local=np.load(folder/"native.npy");edited=old.copy();edited[a:b]=local[a:b]
                context=np.load(checked(candidate["context"]));scaler=np.load(checked(assets["scaler"]));expected=context["values"]*scaler["scale"]+scaler["bias"]
                norm=np.load(folder/"normalized.npy");mask=context["mask"]
                if abs(norm[mask]-expected[mask]).max()>2e-6:raise ValueError("Known inpaint context not restored")
                noise=read(folder/"generation.json")["sampler"]["noise_sha256"]
            if not np.array_equal(edited[protected],old[protected]):raise ValueError("Bridge edited protected motion")
            points,global_physics=audit_native(edited);_,roi_physics=audit_native(edited[:150])
            np.save(folder/"edited_native.npy",edited);np.save(folder/"edited_fk.npy",points);write(folder/"global_physical.json",global_physics);write(folder/"roi_physical.json",roi_physics)
            prior=region(old_fk,cfg["boundary"]-6,cfg["boundary"]+6);after=region(points,cfg["boundary"]-6,cfg["boundary"]+6)
            joins_before=[region(old_fk,t-3,t+3) for t in (a,b)];joins_after=[region(points,t-3,t+3) for t in (a,b)]
            after_source=source_error(edited,plan,a,b)
            ok=after["p95"]<=prior["p95"]*(1-cfg["gate"]["boundary_p95_reduction"]) and all(x["maximum"]<=y["maximum"]+cfg["gate"]["join_jerk_nonregression"] for x,y in zip(joins_after,joins_before)) and after_source<=before_source+cfg["gate"]["source_nonregression_rad"] and roi_physics["absolute"]["ok"]
            row=dict(name=name,arm=arm,noise_sha256=noise,protected_bitwise_equal=True,edit=list(cfg["edit"]),boundary_before=prior,boundary_after=after,joins_before=joins_before,joins_after=joins_after,
                source_error_before_rad=before_source,source_error_after_rad=after_source,roi_native_ok=roi_physics["absolute"]["ok"],global_native_ok=global_physics["absolute"]["ok"],repair_qualified=ok)
            records.append(row);bindings.append(dict(name=name,arm=arm,motion=ref(folder/"edited_native.npy"),fk=ref(folder/"edited_fk.npy"),global_physical=ref(folder/"global_physical.json"),roi_physical=ref(folder/"roi_physical.json")))
            if replay_from:
                previous=Path(replay_from)/folder.name
                for file in ("edited_native.npy","edited_fk.npy"):
                    if not np.array_equal(np.load(folder/file),np.load(previous/file)):raise ValueError("Protected global repair replay differs")
                for file in ("global_physical.json","roi_physical.json"):
                    if read(folder/file)!=read(previous/file):raise ValueError("Repair Native replay differs")
                if arm!="smooth":
                    for file in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy"):
                        if not np.array_equal(np.load(folder/file),np.load(previous/file)):raise ValueError("Bridge sampling replay differs")
            print("R8_BRIDGE",name,arm,ok,flush=True)
    qualified=[r for r in records if r["arm"]=="event_inpaint"]
    for candidate in candidates:
        noise={r["noise_sha256"] for r in records if r["name"]==candidate["name"] and r["arm"]!="smooth"}
        if len(noise)>1:raise ValueError("Inpaint/source-inpaint noise differs")
    result=dict(records=records,rejected=reject,bridge_gate_passed=len(qualified)==len(cfg["cases"]) and all(r["repair_qualified"] for r in qualified),
        new_local_generations=sum(r["arm"]!="smooth" for r in records),new_whole_song_generations=0,neural_weight_updates=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,reperformance_config=rconfig,audit=ref(Path(audit)/"protocol.json"),bindings=bindings))
    if replay_from:
        if read(Path(replay_from)/"result.json")!=result:raise ValueError("Bridge readout replay differs")
        write(out/"replay.json",dict(status="FRESH_BIDIRECTIONAL_INPAINT_AND_GLOBAL_NATIVE_REPLAY",new_local_generations=result["new_local_generations"],protected_motion_exact=True))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--reperformance-config",required=True);p.add_argument("--audit",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),read(a.reperformance_config),a.audit,a.out,a.replay_from)

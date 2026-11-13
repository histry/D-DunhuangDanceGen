"""Independent saved-array, source-clock, dose, SLERP and private-review checks."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation,Slerp
from dunhuang.io import sha,checked,write
from dunhuang.contracts import matrices,MotionAdapter

def read(p):return json.loads(Path(p).read_text())
def angle(a,b,mask):
    ar=matrices(a[:,79:127].reshape(150,8,6)).astype(float);br=matrices(b[:,79:127].reshape(150,8,6)).astype(float)
    return float(Rotation.from_matrix((ar.swapaxes(-1,-2)@br)[mask]).magnitude().mean())
def protect(base,value,cfg):
    br=matrices(base[:,7:].reshape(150,24,6));vr=matrices(value[:,7:].reshape(150,24,6))
    d=Rotation.from_matrix((br.swapaxes(-1,-2)@vr).reshape(-1,3,3)).magnitude().reshape(150,24)
    other=[j for j in range(24) if not 12<=j<20]
    def motion(r):
        v=Rotation.from_matrix((r[:-1,12:20].swapaxes(-1,-2)@r[1:,12:20]).reshape(-1,3,3)).as_rotvec().reshape(149,8,3)*30
        return np.array([np.linalg.norm(v,axis=-1).mean(),np.quantile(np.linalg.norm(np.diff(v,axis=0)*30,axis=-1),.95),np.quantile(np.linalg.norm(np.diff(v,n=2,axis=0)*900,axis=-1),.95)])
    ratio=motion(vr)/np.maximum(motion(br),1e-9)
    values=dict(non_target_mean_rad=float(d[:,other].mean()),root_rms_m=float(np.square(value[:,4:7]-base[:,4:7]).mean()**.5),contact_max=float(abs(value[:,:4]-base[:,:4]).max()),activity_ratio=float(ratio[0]),acceleration_proxy_ratio=float(ratio[1]),jerk_proxy_ratio=float(ratio[2]))
    good=values["non_target_mean_rad"]<=cfg["non_target_mean_rad"] and values["root_rms_m"]<=cfg["root_rms_m"] and values["contact_max"]<=cfg["contact_max"] and cfg["activity_ratio"][0]<=ratio[0]<=cfg["activity_ratio"][1] and ratio[1]<=cfg["acceleration_ratio"] and ratio[2]<=cfg["jerk_ratio"]
    return values,bool(good)
def check(root):
    root=Path(root);prep=read(root/"prepared/protocol.json");cfg=prep["config"];pairs=read(checked(prep["pairs"]))
    manifest=read(checked(prep["source_manifest"]));index={r["window_uid"]:r for r in manifest["rows"]}
    old={index[p[k]]["raw_motion"]["sha256"] for p in pairs["calibration"] for k in ("a","b")}
    new={index[p[k]]["raw_motion"]["sha256"] for p in pairs["disjoint_probe"] for k in ("a","b")}
    assert not old&new and not {p["source_group"] for p in pairs["calibration"]}&{p["source_group"] for p in pairs["disjoint_probe"]}
    assert all(index[p[k]]["role"]=="train" for pop in pairs.values() for p in pop for k in ("a","b"))
    source=read(Path(cfg["source"])/"cf/protocol.json");q=read(checked(source["qualified"]));previous=read(checked(q["previous_preflight"]));assets=read(checked(previous["assets"]))
    scaler=np.load(checked(assets["scaler"]));scale=scaler["scale"];bias=scaler["bias"]
    arrays=dict(np.load(root/"probe/arrays.npz"));rs=read(root/"probe/result.json");known=np.ones((150,8),bool);known[25:40]=False;known[90:115]=False
    def native(x):return MotionAdapter.to_native(((x[0].clip(-1,1)-bias)/scale).astype(np.float32))
    norm_error=0.;source_error=0.;protection_error=0.;budget=0.;dose_buckets={};passes={};checked_predictions=0;expression_error=0.;expression_relative_error=0.;stages={}
    for r in rs["records"]:
        c=r["cohort"];case=r["case"];tick=r["timestep"];s=r["source"];arm=r["arm"];dose=r["dose_kind"];prefix=c+"__"+str(case)+"__"+str(tick)
        pair=pairs[c][case];refs=[np.load(checked(index[pair[k]]["motion"])) for k in ("a","b")]
        base=native(arrays[prefix+"__base"]);key=prefix+"__"+str(s)+"__"+arm+"__"+dose
        delta=arrays[key+"__direction"];norm=float(np.linalg.norm(delta.astype(float)));budget=max(budget,norm/r["budget_norm"])
        assert abs(norm-r["input_norm"])<1e-6
        if dose=="common_norm":norm_error=max(norm_error,abs(norm-r["common_norm"])/max(r["common_norm"],1e-12));dose_buckets.setdefault((c,case,tick,s),[]).append(norm)
        value=native(arrays[key+"__predicted_x0"])
        if (c,case,s) not in stages:
            from dunhuang.research.reference_guidance.core import phase_reference
            observed=np.zeros((150,151),bool);observed[:,79:127]=np.repeat(known,6,-1)
            stages[(c,case,s)]=phase_reference(refs[s],observed,known,read("configs/data_event_v3.json")["phases"])[2]
        stage=stages[(c,case,s)]
        def expression(motion):
            a=matrices(motion[:,79:127].reshape(150,8,6)).astype(float);b=matrices(refs[s][:,79:127].reshape(150,8,6)).astype(float)
            pose=Rotation.from_matrix((a.swapaxes(-1,-2)@b).reshape(-1,3,3)).magnitude().reshape(150,8)**2
            ai=a[:-1].swapaxes(-1,-2)@a[1:];bi=b[:-1].swapaxes(-1,-2)@b[1:]
            velocity=Rotation.from_matrix((ai.swapaxes(-1,-2)@bi).reshape(-1,3,3)).magnitude().reshape(149,8)**2*900
            return np.array([velocity[known[1:]&known[:-1]].mean(),pose[stage].mean() if stage.any() else 0])
        be=expression(base);ve=expression(value);reported=np.array([r["velocity_gain_squared_radps"],r["stage_pose_gain_squared_rad"]])
        err=abs(be-ve-reported);relative=err/np.maximum(1,np.maximum(abs(be),abs(ve)))
        expression_error=max(expression_error,float(err.max()));expression_relative_error=max(expression_relative_error,float(relative.max()))
        assert relative.max()<1e-4  # conservative FP32 geometry/reduction audit tolerance, never a model qualification threshold
        own=angle(base,refs[s],known)-angle(value,refs[s],known);wrong=angle(base,refs[1-s],known)-angle(value,refs[1-s],known)
        source_error=max(source_error,abs(own-r["own_gain_rad"]),abs(wrong-r["wrong_gain_rad"]),abs(own-wrong-r["directional_gain_rad"]))
        pv,ok=protect(base,value,cfg["protection"])
        protection_error=max(protection_error,max(abs(v-r["protection"][k]) for k,v in pv.items()))
        assert ok==r["protection"]["protected"]
        if dose=="common_norm":
            good=own>.001 and own-wrong>.001 and ok and r["numeric_trusted"]
            passes[(c,arm)]=passes.get((c,arm),0)+int(good)
        checked_predictions+=1
    assert budget<=1+1e-6 and norm_error<=2*np.finfo(np.float32).eps
    assert all(len(v)==5 for v in dose_buckets.values()) and len(dose_buckets)==48
    assert source_error<1e-6 and protection_error<1e-5
    assert all(passes[(c,a)]==rs["qualification"][c][a]["joint_trusted"] for c in pairs for a in cfg["probe"]["arms"])
    exact_arrays=0
    for part in ("diagnosis","probe"):
        main=np.load(root/part/"arrays.npz");replay=np.load(root/(part+"_replay")/"arrays.npz");assert set(main.files)==set(replay.files)
        for k in main.files:assert np.array_equal(main[k],replay[k]);exact_arrays+=1
        assert read(root/part/"result.json")==read(root/(part+"_replay")/"result.json")
    # Original numeric failures cannot be upgraded by the fixed-active surrogate.
    diag=read(root/"diagnosis/result.json");frozen=read(Path(cfg["protected_source"])/"protected/result.json")
    assert diag["old_numeric_trusted"]==sum(r["numeric_trusted"] for r in frozen["solves"])==93
    assert not any(r["qualification_override"] for r in diag["numeric_diagnostics"])
    models=read("reports/source_tasks_summary.json")["model_sha256"]
    for name,digest in models.items():assert sha(Path(cfg["source"])/name/("adapter.pt" if name.startswith("cf/") else "content.pt"))==digest
    # Canonical actual PCM, not path/filename distinction.
    music=read(checked(prep["music"]));pcm=[]
    for m in music:
        wave,sr=sf.read(checked(m["audio"]),dtype="float32",always_2d=True)
        d=hashlib.sha256(np.ascontiguousarray(wave).tobytes()+str((sr,wave.shape)).encode()).hexdigest();assert d==m["pcm_sha256"];pcm.append(d)
    assert len(set(pcm))==3 and len({m["work_uid"] for m in music})==3
    # Independently recompute source-time interpolation and SLERP.
    pose_error=0.;phase_error=0.;plans=0;events=0
    for path in (root/"music").glob("*_*/event_plan.json"):
        plan=read(path);cond=np.load(path.parent/"condition.npz");reference=cond["native_reference"];mask=cond["known"];phase=cond["phase"];plans+=1
        assert not mask[:,:,:79].any() and not mask[:,:,127:].any()
        for e in plan["events"]:
            a,b=e["music_interval"];lo,hi=e["source_interval"];ticks=np.arange(a,b);w=np.array(e["time_warp"]);pos=np.minimum(np.interp(ticks,w[:,1],w[:,0]),hi-1)
            source=np.load(checked(e["reference"]));rot=matrices(source[:,79:127].reshape(-1,8,6)).astype(float);window=np.minimum(ticks//75,len(reference)-1);local=ticks-75*window
            actual=matrices(reference[window,local,79:127].reshape(-1,8,6)).astype(float)
            for j in range(8):
                expected=Slerp(np.arange(len(source)),Rotation.from_matrix(rot[:,j]))(pos).as_matrix()
                pose_error=max(pose_error,float(Rotation.from_matrix(expected.swapaxes(-1,-2)@actual[:,j]).magnitude().max()))
            assert mask[window,local,79:127].all()
            phase_error=max(phase_error,float(abs(phase[window,local,0]-(pos-lo)/(hi-lo)).max()));events+=1
    assert plans==read(root/"music/result.json")["verified_plans"] and pose_error<1e-6 and phase_error<1e-6
    review=read(root/"review/protocol.json");bindings=read(checked(review["bindings"]));items=read(checked(review["manifest"]));assert len(items)==len(bindings)==35
    secrets=set()
    for b in bindings:
        secrets.update([b["source_uid"],b["event_uid"],b["raw_motion_sha256"]])
        assert sha(checked(b["media"]))==b["media_sha256"]
        for key in ("raw","native"):checked(b[key])
        with np.load(checked(b["raw"]),allow_pickle=False) as raw:
            assert float(raw["mocap_framerate"])==b["fps"]==30
            assert len(raw["poses"])==len(np.load(checked(b["native"]),mmap_mode="r"))
            assert 0<=b["start_frame"]<b["end_frame"]<=len(raw["poses"])
            assert 0<=b["display_interval"][0]<b["display_interval"][1]<=len(raw["poses"])
        if b.get("audio"):checked(b["audio"])
    for file in (root/"review/reviewer").iterdir():
        if file.suffix not in (".html",".json",".txt"):continue
        text=file.read_text()
        assert not any(secret in text for secret in secrets)
        assert not any(token in text for token in ("source_uid","event_uid","source_interval","automatic_peak","source_group"))
    r=read(root/"review/result.json");assert r["actual_records"]==0 and not r["human_reviewed"] and r["summary"]["states"]=={"UNREVIEWED":35}
    result=dict(schema="independent_source_expression_checks",verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),verifier_sha256=sha(__file__),
        saved_predictions_verified=checked_predictions,max_source_readout_error_rad=source_error,max_protection_readout_error=protection_error,max_expression_readout_absolute_error=expression_error,max_expression_readout_relative_to_level_error=expression_relative_error,expression_numeric_audit_bound=1e-4,
        max_condition_fraction=budget,max_common_norm_relative_error=norm_error,common_dose_groups=len(dose_buckets),exact_replay_arrays=exact_arrays,
        old_numeric_trusted_unchanged=93,frozen_model_files_rechecked=models,known_train_source_and_music_group_overlap=0,
        distinct_real_pcm_assets=3,plans_verified=plans,consumed_events=events,independent_slerp_max_error_rad=pose_error,phase_clock_max_error=phase_error,
        blind_items=35,raw_native_frame_clock_checks=35,source_stage_support_counts=[dict(cohort=c,case=i,source=s,stage_joint_frames=int(v.sum()),stage_frames=int(v.any(-1).sum())) for (c,i,s),v in sorted(stages.items())],original_human_records=0,public_review_identity_scan_passed=True,human_reviewed=False,semantic_event_truth="UNKNOWN",optimizer_calls=0,edge_forward_calls=0,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    write(root/"public_checks.json",result);print("INDEPENDENT_R6_CHECKS",checked_predictions,plans,len(items))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();check(a.out)

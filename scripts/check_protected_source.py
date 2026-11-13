"""Independent saved-output, neighbour-bank, relabel and source-consumption checks."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from dunhuang.io import sha,checked,write
from dunhuang.contracts import matrices
def read(p):return json.loads(Path(p).read_text())
def angle(a,b,mask):
    ar=matrices(a[:,79:127].reshape(150,8,6)).astype(float);br=matrices(b[:,79:127].reshape(150,8,6)).astype(float)
    return float(Rotation.from_matrix((ar.swapaxes(-1,-2)@br)[mask]).magnitude().mean())
def metric(y,p):
    y=np.asarray(y,bool);order=np.argsort(-p,kind="stable");truth=y[order];ranked=p[order]
    if not y.any() or y.all():return None
    end=np.r_[np.flatnonzero(ranked[1:]!=ranked[:-1]),len(y)-1];tp=truth.cumsum()[end]
    ap=float((np.diff(np.r_[0,tp])/truth.sum()*tp/(end+1)).sum());predict=p>=.5
    return np.array([ap,float(2*np.sum(predict&y)/max(int(predict.sum()+y.sum()),1))])
def aggregate(y,p,groups,recordings):
    result={}
    for g in sorted(set(groups)):
        items=[metric(y[recordings==r],p[recordings==r]) for r in sorted(set(recordings[groups==g]))]
        items=[v for v in items if v is not None]
        if items:result[g]=np.mean(items,0)
    return result
def check(root):
    root=Path(root);c=read(root/"protected/result.json");e=read(root/"information/result.json");pp=read(root/"planner/result.json")
    arrays=dict(np.load(root/"protected/arrays.npz"));control_cfg=read(root/"protected/protocol.json")["config"];control=Path(control_cfg["reference_control"])
    old=dict(np.load(control/"control/arrays.npz"));errors=[];budget=0.;matched=0.;baselines=0
    for scope in ("train","val"):
        for case in range(4):
            for tick in (999,499,19):
                label=scope+"__"+str(case)+"__"+str(tick)
                assert np.array_equal(arrays[label+"__base"],old[label+"__base"])
                for source in (0,1):
                    take=[r for r in c["records"] if r["scope"]==scope and r["case"]==case and r["timestep"]==tick and r["source"]==source]
                    values={r["arm"]:r for r in take};assert len(values)==8
                    for arm,r in values.items():
                        delta=arrays[label+"__"+str(source)+"__"+arm+"__direction"];norm=float(np.linalg.norm(delta.astype(float)))
                        assert abs(norm-r["input_norm"])<1e-6
                        budget=max(budget,norm/r["budget_norm"])
                        if arm in ("B1","B2"):
                            assert np.array_equal(arrays[label+"__"+str(source)+"__"+arm+"__predicted_x0"],old[label+"__"+str(source)+"__"+arm+"__predicted_x0"]);baselines+=1
                    for suffix in ("1","2"):
                        norm=values["PB"+suffix]["input_norm"]
                        for arm in ("shrink_B"+suffix,"random_PB"+suffix):matched=max(matched,abs(norm-values[arm]["input_norm"]))
    assert budget<=1+1e-6 and matched<1e-6 and baselines==96
    cf=read("outputs/source_tasks_02/cf/protocol.json");q=read(checked(cf["qualified"]));pre=read(checked(q["previous_preflight"]));manifest=read(checked(pre["manifest"]));index={r["window_uid"]:r for r in manifest["rows"]}
    known=np.ones((150,8),bool);known[25:40]=False;known[90:115]=False
    for case,pair in enumerate(manifest["validation_pairs"]):
        refs=[np.load(checked(index[pair[k]]["motion"])) for k in ("a","b")];base=np.load(root/"protected"/("val__%s__19__base"%case)/"raw_native.npy")
        for r in c["records"]:
            if r["scope"]!="val" or r["case"]!=case or r["timestep"]!=19 or r["arm"] not in ("PB1","PB2"):continue
            value=np.load(root/"protected"/("val__%s__19__%s__%s"%(case,r["source"],r["arm"]))/"raw_native.npy");s=r["source"]
            own=angle(base,refs[s],known)-angle(value,refs[s],known);wrong=angle(base,refs[1-s],known)-angle(value,refs[1-s],known)
            errors.extend([abs(own-r["own_gain_rad"]),abs((own-wrong)-r["directional_gain_rad"])])
    assert max(errors)<1e-6
    data=read("outputs/source_tasks_02/content_data/protocol.json");rows=read(checked(data["manifest"]))["rows"]
    info=dict(np.load(root/"information/arrays.npz"));indices=info["indices"]
    groups=np.array([rows[int(i)]["source_group"] for i,t,f in indices]);recordings=np.array([rows[int(i)]["recording_uid"] for i,t,f in indices]);roles=np.array([rows[int(i)]["role"] for i,t,f in indices]);bank=info["bank"];assert (roles[bank]=="train").all()
    label=info["labels"];count=0;metric_errors=[]
    for scope in ("train_leave_song_out","val_known_exposed_songs"):
        qs=info["train_queries"] if scope.startswith("train") else info["val_queries"];agg={}
        for variant in ("F0_F1","F2","F3","F4_pose","F4_time"):
            ns=info[scope+"__"+variant+"__neighbours"]
            assert np.isin(ns,bank).all() and (groups[qs,None]!=groups[ns]).all()
            for name,column in (("speed_peak",0),("hold_centre",1),("hold_interval",2)):
                if name=="hold_centre" and variant!="F0_F1":continue
                score=label[ns,column].mean(1);assert np.array_equal(score,info[scope+"__"+variant+"__"+name]);count+=1
                agg[(variant,name)]=aggregate(label[qs,column],score,groups[qs],recordings[qs])
        for name in ("speed_peak","hold_interval"):
            for variant in ("F0_F1","F2","F3","F4_pose","F4_time"):
                for key,b in (("vs_F1_dynamic","F0_F1"),("vs_F3","F3")):
                    actual=e["metrics"][scope][name][variant][key]
                    if actual["status"]=="NOT_EVALUABLE":continue
                    a=agg[(variant,name)];b=agg[(b,name)];keys=sorted(set(a)&set(b));av=np.array([a[k] for k in keys]);bv=np.array([b[k] for k in keys]);delta=av-bv
                    vals=np.r_[av.mean(0),delta.mean(0)];ref=np.array([actual["AP"],actual["F1"],actual["delta_AP"],actual["delta_F1"]]);metric_errors.extend(abs(vals-ref).tolist())
                    rng=np.random.default_rng(20261113);draws=delta[rng.integers(0,len(keys),(1000,len(keys)))].mean(1);ci=np.quantile(draws,[.025,.975],axis=0)
                    metric_errors.extend(abs(ci-np.array(actual["delta95"])).flatten().tolist())
    assert max(metric_errors,default=0)<1e-12
    # Re-evaluate saved historical heads under both target definitions, without fitting.
    previous=dict(np.load(Path(control_cfg["reference_events"])/"events/scores.npz"));val=info["val_queries"];frozen_heads={}
    for name in ("simple_time_signal","raw_observations","15","30","dual","dual_shuffle","15_permuted","30_permuted","dual_permuted","dual_shuffle_permuted"):
        score=previous["motion__upper_pose_hold__"+name][val];summary={}
        for target,column in (("centre",1),("interval",2)):
            a=aggregate(label[val,column],score,groups[val],recordings[val])
            summary[target]=dict(groups=len(a),AP=float(np.mean([v[0] for v in a.values()])) if a else None,F1=float(np.mean([v[1] for v in a.values()])) if a else None)
        frozen_heads[name]=summary
    # Analytic kinematic recoverability check, not an independently labelled benchmark.
    from dunhuang.planning.time_map import motion_features74
    from dunhuang.data.change_event_phases import detect
    cfg=read("configs/data_event_v3.json")["phases"]
    buckets={}
    for n,(i,t,f) in enumerate(indices):buckets.setdefault(rows[int(i)]["raw_motion"]["sha256"],[]).append(n)
    direct=np.zeros((len(indices),2),bool)
    for asset,ids in sorted(buckets.items()):
        row=rows[int(indices[ids[0],0])];native_source=np.load(checked(row["source_native"]))
        features,observed=motion_features74(native_source);omega=features[:,:66].reshape(-1,22,3)[:,12:20]
        speed=np.linalg.norm(omega,axis=-1).mean(1);vec=omega.reshape(len(speed),-1);norm=np.linalg.norm(vec,axis=-1)
        cosine=(vec[1:]*vec[:-1]).sum(-1)/np.maximum(norm[1:]*norm[:-1],1e-12)
        events=detect(dict(upper8=speed,upper_direction_change=np.r_[0,1-np.clip(cosine,-1,1)],root_yaw_radps=np.full(len(speed),np.nan)),cfg)
        frames=indices[ids,2]
        for event in events:
            if event["type"]=="upper_pose_hold":
                lo,hi=event["evidence_span"];direct[ids,1]|=(frames>=lo)&(frames<hi)
            if event["type"]=="upper_speed_peak":direct[ids,0]|=abs(frames-event["frame"])<=2
    reconstruction={}
    for name,column,truth_column in (("speed_peak",0,0),("hold_interval",1,2)):
        truth=label[:,truth_column];pred=direct[:,column]
        tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum());tn=int((~truth&~pred).sum())
        by_role={}
        for role in ("train","val"):
            take=roles==role;t=truth[take];v=pred[take];rtp=int((t&v).sum());rfp=int((~t&v).sum());rfn=int((t&~v).sum())
            by_role[role]=dict(F1=2*rtp/max(2*rtp+rfp+rfn,1),agreement=float((t==v).mean()))
        reconstruction[name]=dict(TP=tp,FP=fp,FN=fn,TN=tn,F1=2*tp/max(2*tp+fp+fn,1),agreement=float((truth==pred).mean()),by_role=by_role,
            rule_shared_with_proxy_detector=True,semantic_ground_truth=False)
    data_root=Path(control_cfg["data_v3"]);parent=read(data_root/"main/protocol.json")
    catalog=read(checked(parent["catalog"]))["entries"];event_index={v["event_uid"]:i for i,v in enumerate(catalog)}
    new={v["event_uid"]:v for v in read(data_root/"main/event_catalog_v2_shadow.json")["entries"]}
    slot_index={(r["timeline_identity"],r["slot"]["start"],r["slot"]["end"]):i for i,r in enumerate(read(data_root/"main/music_slots.json"))}
    gold=dict(np.load(Path(control_cfg["warp"])/"main/maps.npz"))
    max_pose=0.;phase_error=0.;plan_count=0;candidate_count=0;geometry_shapes=[];music_hashes=set()
    import torch
    from dunhuang.generation.event_geometry import event_geometry
    from dunhuang.generation.event_visibility import clean_geometry
    torch.set_num_threads(4)
    for plan in (root/"planner").glob("*_*/event_plan.json"):
        case=int(plan.parent.name.split("_",1)[0]);arm=plan.parent.name.split("_",1)[1];arm_index={"E0_legacy":0,"E1_legacy":1,"E1_W":3}[arm]
        value=read(plan);music_hashes.add(value["audio_sha256"]);condition=dict(np.load(plan.parent/"condition.npz"));native=condition["native_reference"];mask=condition["known"];phase=condition["phase"];frames=value["frames"]
        if mask[:,:,:79].any() or mask[:,:,127:].any():raise ValueError("Unknown Root/contact became a source condition")
        with torch.no_grad():
            values=torch.from_numpy(native.copy());known=torch.from_numpy(mask.copy());phases=torch.from_numpy(phase.copy())
            geometry=event_geometry(values,known,phases);values[~known]=float("nan")
            poisoned=event_geometry(values,known,phases)
            if not torch.equal(geometry,poisoned):raise ValueError("Actual plan geometry reads unknown storage")
            clean,valid=clean_geometry(geometry)
            if clean.shape!=(len(native),150,153) or not torch.isfinite(clean).all():raise ValueError("Observed event interface differs")
            if not torch.equal(valid,known[:,:,79:127].any(-1)):raise ValueError("Observed event support changed")
            geometry_shapes.append(list(clean.shape))
        for event in value["events"]:
            payload={k:event[k] for k in ("event_uid","source_interval","music_interval","time_warp","reference")}
            if event.get("root_observation"):payload["root_observation"]=event["root_observation"]
            payload.update(expression=event["expression"],condition_schema="soft_event_reference_v1",score_models=None)
            assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()==event["candidate_uid"]
            candidate_count+=1
            a,b=event["music_interval"]
            row=slot_index[(parent["timelines"][case]["semantic_identity"],a,b)];col=event_index[event["event_uid"]]
            if not np.array_equal(np.asarray(event["time_warp"]),gold["warps"][arm_index,row,col]):raise ValueError("Consumed map differs from frozen E1-W mathematical certificate")
            expected_center=catalog[col]["center"] if arm=="E0_legacy" else new[event["event_uid"]]["safe_anchor"]["frame"]
            if event["primitive"]["source_frame"]!=expected_center:raise ValueError("Actual source anchor changed")
            a,b=event["music_interval"];lo,hi=event["source_interval"];ticks=np.arange(a,b);warp=np.array(event["time_warp"])
            pos=np.minimum(np.interp(ticks,warp[:,1],warp[:,0]),hi-1);source=np.load(checked(event["reference"]));rot=matrices(source[:,79:127].reshape(-1,8,6)).astype(float)
            window=np.minimum(ticks//75,len(native)-1);local=ticks-75*window
            actual=matrices(native[window,local,79:127].reshape(-1,8,6)).astype(float)
            for joint in range(8):
                # Independent whole-source SLERP, not the production resample helper.
                expected=Slerp(np.arange(len(source)),Rotation.from_matrix(rot[:,joint]))(pos).as_matrix()
                d=Rotation.from_matrix(expected.swapaxes(-1,-2)@actual[:,joint]).magnitude();max_pose=max(max_pose,float(d.max()))
            assert mask[window,local,79:127].all()
            phase_error=max(phase_error,float(abs(phase[window,local,0]-(pos-lo)/(hi-lo)).max()))
        plan_count+=1
    assert plan_count==pp["verified_plans"] and plan_count>0
    assert max_pose<1e-6 and phase_error<1e-6
    frozen=read("reports/source_tasks_summary.json");models={}
    for name,expected in frozen["model_sha256"].items():
        file=Path("outputs/source_tasks_02")/name/("adapter.pt" if name.startswith("cf/") else "content.pt");assert sha(file)==expected;models[name]=expected
    checks=dict(schema="independent_protected_source_checks",verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),verifier_sha256=sha(__file__),
        max_condition_fraction=budget,max_matched_norm_difference=matched,old_B1_B2_predictions_exact=baselines,source_readouts_verified=len(errors)//2,max_source_readout_error_rad=max(errors),
        neighbour_score_arrays_verified=count,frame_metric_values_verified=len(metric_errors),max_frame_metric_error=max(metric_errors,default=0),all_banks_train_only=True,all_queries_music_disjoint_from_neighbours=True,
        frozen_historical_head_relabel=frozen_heads,direct_74_kinematic_reconstruction=reconstruction,head_fit_calls=0,normalizer_fits=0,plans_verified=plan_count,candidate_identities_verified=candidate_count,frozen_selected_map_and_anchor_parities=candidate_count,
        distinct_shadow_audio_assets=len(music_hashes),observed_geometry_shapes=geometry_shapes,actual_unknown_poison_cases_verified=len(geometry_shapes),observed_geometry_is_not_production_weight_qualification=True,independent_slerp_max_error_rad=max_pose,phase_clock_max_error=phase_error,frozen_model_files_rechecked=models,optimizer_calls=0,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    write(root/"public_checks.json",checks);print("INDEPENDENT_PROTECTED_CHECKS",baselines,count,plan_count)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();check(a.out)

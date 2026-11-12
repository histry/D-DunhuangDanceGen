"""Data Split V3 + ChangE-E1: offline audit, real source replay, zero model fitting."""
import argparse,collections,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,sha,pinned
from dunhuang.data.sources import ref
from dunhuang.data.split_registry import stable,catalog_subset
from dunhuang.data.change_event_phases import motion_profiles,detect
from dunhuang.data.change_event_audit import audit_event,slot_eligibility,summarize
from dunhuang.planning.time_map import motion_features74
from .registry import build as registry_build

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def register_timelines(output_root,frame_counts):
    choices={}
    for path in sorted(Path(output_root).glob("**/music_timeline.json")):
        companion=path.parent/"event_plan.json"
        if not companion.exists():continue
        t=read(path);plan=read(companion)
        if t.get("fps")!=30 or t.get("frames") not in frame_counts or plan.get("schema")!="event_plan_v1":continue
        cursor=0
        for s in t["slots"]:
            if type(s["start"]) is not int or type(s["end"]) is not int or s["start"]!=cursor or s["end"]<=cursor:raise ValueError("Frozen timeline is not contiguous")
            cursor=s["end"]
        if cursor!=t["frames"]:raise ValueError("Frozen timeline duration differs")
        payload=dict(audio_sha256=plan["audio_sha256"],fps=30,frames=t["frames"],beats=t["beats"],slots=t["slots"])
        key=stable(payload);entry=dict(timeline=ref(path),plan=ref(companion),semantic_identity=key,payload=payload)
        current=choices.get(t["frames"])
        if current is None or (key,str(path))<(current["semantic_identity"],current["timeline"]["path"]):choices[t["frames"]]=entry
    if set(choices)!=set(frame_counts):raise ValueError("Registered real timeline frame populations unavailable")
    return [choices[k] for k in frame_counts]

def readouts_by_source(diagnostics):
    result=[]
    for uid in sorted({r["source_uid"] for r in diagnostics}):
        rows=[r for r in diagnostics if r["source_uid"]==uid]
        result.append(dict(source_uid=uid,events=len(rows),old_clamps=sum(r["old_peak_clamped"] for r in rows),
            e0_peak_coverage=sum(r["e0_captures_controlled_peak"] for r in rows),e1_peak_coverage=sum(r["e1_captures_controlled_peak"] for r in rows),
            e0_support=sum(r["e0_normalized_support"] for r in rows),e1_support=sum(r["e1_normalized_support"] for r in rows),
            no_safe_anchor=sum(not r["e1_source_context"] for r in rows)))
    return result

def run(assets_file,population,config,out,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    if config["optimizer_calls"] or config["statistics_fit_calls"] or config["allow_generation"]:raise ValueError("This registered stage is offline only")
    assets=read(assets_file);catalog=read(checked(assets["catalog"]));upstream=read(checked(assets["catalog_source"]))
    if catalog["schema"]!="event_catalog_v1" or len(catalog["entries"])!=config["expected_events"]:raise ValueError("Frozen event population differs")
    if sha(checked(assets["catalog_source"]))!=catalog["source_catalog_sha256"]:raise ValueError("Upstream Catalog binding differs")
    if catalog["scaler_sha256"]!=assets["scaler"]["sha256"]:raise ValueError("Frozen Catalog scaler differs")
    timelines=register_timelines("outputs",config["timeline_frames"])
    protocol=dict(schema="data_event_v3_e1_registered_study",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config=config,assets=ref(assets_file),catalog=assets["catalog"],source_catalog=assets["catalog_source"],scaler=assets["scaler"],
        population=ref(population),timelines=timelines,source_normalization_refitted=False,
        optimizer_calls=0,statistics_fit_calls=0,edge_forward_calls=0,native_calls=0,operational_qualified=False)
    write(out/"protocol.json",protocol)
    registry,attestation,exposure=registry_build(assets,population,config,out)
    if attestation["forbidden_identity_overlaps"]:raise ValueError("Forbidden task identities cross roles")
    with np.load(checked(assets["scaler"])) as z:scale,bias=z["scale"],z["bias"]
    legacy=pinned(assets["legacy_root"],assets["legacy_commit"]);sys.path.insert(0,str(legacy))
    from motion.retrieval_generation import decode_continuous_source
    from motion.geometry import fk_24_np
    native_dir=out/"native";native_dir.mkdir();raw_dir=out/"raw";raw_dir.mkdir();feature_dir=out/"features";feature_dir.mkdir()
    sources={};source_profiles={};source_phases={};source_seals=[]
    upstream_events={e["event_uid"]:e for e in upstream["events"]}
    references={}
    for e in catalog["entries"]:
        if e["source_uid"] in references and references[e["source_uid"]]!=e["reference"]:raise ValueError("Conflicting full-record references")
        references[e["source_uid"]]=e["reference"]
    for source in upstream["source_records"]:
        uid=source["source_uid"];meta=source["discovery_metadata"]
        if meta["recording_performer_count"]!=1 or meta["solo_compatible"] is not True:raise ValueError("E1 single-person source missing")
        if meta["coordinate_system"]!="y_up" or meta["translation_units"]!="m":raise ValueError("Source metric Root/FK contract incomplete")
        original=Path(assets["motion_root"])/Path(source["raw_receipt"]["path"]).name
        checked(dict(path=str(original),sha256=source["source_sha256"]))
        copy_root=raw_dir/source["source_sha256"];copy_root.mkdir();copy=copy_root/original.name;shutil.copyfile(original,copy);checked(dict(path=str(copy),sha256=source["source_sha256"]))
        n=source["raw_frames"];native,_=decode_continuous_source(copy,meta,(0,n),known_rows=np.ones(n,bool))
        frozen=np.load(checked(references[uid]),allow_pickle=False)
        if not np.array_equal(native,frozen):raise ValueError("Fresh raw ChangE decode differs from original full source")
        file=native_dir/(source["source_sha256"]+".npy");np.save(file,native)
        body_counts={upstream_events[e["event_uid"]]["adapter_report"]["observed_body_joint_count"] for e in catalog["entries"] if e["source_uid"]==uid}
        if len(body_counts)!=1:raise ValueError("Observed body support differs within recording")
        observed=list(range(next(iter(body_counts))))
        fk=fk_24_np(native)
        profiles=motion_profiles(native,observed,fk,root_observed=True)
        phases=detect(profiles,config["phases"])
        np.savez_compressed(feature_dir/(source["source_sha256"]+".profiles.npz"),**profiles)
        write(feature_dir/(source["source_sha256"]+".phases.json"),phases)
        sources[uid]=(native,observed,ref(file),source);source_profiles[uid]=profiles;source_phases[uid]=phases
        source_seals.append(dict(source_uid=uid,raw_sha256=source["source_sha256"],copied_raw=ref(copy),native_sha256=sha(file),
            original_native=references[uid],source_metadata_digest=stable(meta),discovery_read_support=[0,len(native)],observed_joints=observed))
        print("E1_SOURCE_DECODED",len(sources),len(upstream["source_records"]),flush=True)
    if len(sources)!=config["expected_sources"]:raise ValueError("Frozen source population differs")
    events=[];diagnostics=[];feature_banks={}
    for old in catalog["entries"]:
        uid=old["source_uid"];native,observed,native_ref,source=sources[uid];u=upstream_events[old["event_uid"]]
        if [u["core_frame_start"],u["core_frame_end"]]!=old["source_interval"]:raise ValueError("Frozen event bounds differ")
        e,d=audit_event(old,native,source_profiles[uid],source_phases[uid],observed,scale,bias,assets["catalog"]["sha256"],u)
        e["reference"]=native_ref
        a,b=e["source_interval"]
        # Versioned source feature index: no latent encoder, no normalization fit.
        if uid not in feature_banks:feature_banks[uid]=motion_features74(native)
        features,mask=feature_banks[uid]
        feature_file=feature_dir/(e["variant_uid"].split(":")[1]+".npz")
        np.savez_compressed(feature_file,motion74=features[a:b],observed74=mask[a:b])
        e["motion_features"]=ref(feature_file);e["feature_schema"]="native30_whole_record_derivatives_before_core_v3"
        e["motif_cluster_uid"]="UNKNOWN_NOT_FITTED";events.append(e);diagnostics.append(d)
    if {e["event_uid"] for e in events}!={e["event_uid"] for e in catalog["entries"]}:raise ValueError("Historical event UID population changed")
    slots=[];matrices={k:[] for k in ("e0_temporal","e1_temporal","e0_adapter","e1_adapter")};reasons={k:collections.Counter() for k in matrices}
    for timeline in timelines:
        t=timeline["payload"]
        for index,slot in enumerate(t["slots"]):
            slots.append(dict(timeline_identity=timeline["semantic_identity"],frames=t["frames"],slot_index=index,slot=slot))
            for key in matrices:
                version,mode=key.split("_");row=[]
                for old,new,d in zip(catalog["entries"],events,diagnostics):
                    entry=dict(source_interval=old["source_interval"],center=old["center"] if version=="e0" else None if new["safe_anchor"] is None else new["safe_anchor"]["frame"],
                        normalization_support=d["e0_normalized_support"] if version=="e0" else new["normalization_support"])
                    good,reason,warp=slot_eligibility(entry,slot,t["beats"],mode)
                    row.append(good);reasons[key][reason]+=1
                matrices[key].append(row)
    matrices={k:np.asarray(v,bool) for k,v in matrices.items()}
    np.savez_compressed(out/"planner_eligibility.npz",**matrices)
    report=summarize(diagnostics,matrices,slots);report["planner_rejection_reasons"]={k:dict(v) for k,v in reasons.items()}
    report["per_source"]=readouts_by_source(diagnostics)
    report.update(schema="change_event_e1_audit_v1",baseline_centers_replayed=True,baseline_support_replayed=True,
        all_original_event_uids_preserved=True,all_original_core_bounds_preserved=True,raw_source_decodes_match_v1=True,
        raw_source_copies=len(source_seals),complete_record_discovery_dependencies_preserved=True,
        effective_semantic_motif_count=None,latent_encoder_forward_calls=0,model_normalization_refits=0,
        g2_qualification=read("reports/source_id_03_receipt.json")["source_following"]["g2_2"]["following_case_passes"],
        g3_started=False,phase_match_training_started=False)
    shadow=dict(schema="change_event_catalog_v2_shadow",fps=30,entries=events,
        baseline_catalog_sha256=assets["catalog"]["sha256"],source_registry_identity=registry["identity_digest"],
        segmentation_version="UNCHANGED_V1",center_version="UPPER8_ACTUAL_PHASE_E1",source_only=True,can_route=False,
        full_record_discovery_dependencies=True,model_training_qualified=False,operational_qualified=False)
    folds_dir=out/"fold_indexes";folds_dir.mkdir();fold_index_counts=[]
    for fold in registry["protocols"]["ChangE_Source_Heldout"]["folds"]:
        counts={}
        for role in ("train","val","test"):
            index=catalog_subset(shadow,fold["roles"],role)
            if not index["entries"]:raise ValueError("Fold source identity does not resolve Catalog entries")
            index.update(schema="change_v3_source_filtered_feature_index",fold=fold["fold"],role=role,
                learned_embeddings_built=False,train_only_statistics_pending=True)
            write(folds_dir/("fold_"+str(fold["fold"])+"_"+role+".json"),index)
            counts[role]=len(index["entries"])
        if sum(counts.values())!=len(events):raise ValueError("Fold does not partition frozen event population")
        fold_index_counts.append(dict(fold=fold["fold"],role_events=counts))
    report["source_filtered_fold_indexes"]=fold_index_counts
    write(out/"event_catalog_v2_shadow.json",shadow);write(out/"change_event_e1_report.json",report)
    write(out/"event_diagnostics.json",diagnostics);write(out/"source_seals.json",source_seals);write(out/"music_slots.json",slots)
    result=dict(status="COMPLETED_DATA_V3_CHANGE_E1_SHADOW",registry_identity=registry["identity_digest"],
        data=exposure,split=attestation,e1={k:v for k,v in report.items() if k!="per_source"},
        optimizer_calls=0,statistics_fit_calls=0,edge_forward_calls=0,native_calls=0,operational_qualified=False)
    write(out/"result.json",result)
    if replay_from:
        original=Path(replay_from)
        for name in ("result.json","change_event_e1_report.json","split_attestation_v3.json","data_exposure_audit.json","event_diagnostics.json","music_slots.json"):
            if read(original/name)!=read(out/name):raise ValueError("Fresh V3/E1 scientific readout differs: "+name)
        def semantic(catalog):
            return [dict({k:v for k,v in e.items() if k not in ("reference","motion_features")},reference_sha256=e["reference"]["sha256"],motion_features_sha256=e["motion_features"]["sha256"]) for e in catalog["entries"]]
        if semantic(read(original/"event_catalog_v2_shadow.json"))!=semantic(shadow):raise ValueError("Fresh primitive identity differs")
        for path in (original/"fold_indexes").glob("*.json"):
            target=out/"fold_indexes"/path.name
            if semantic(read(path))!=semantic(read(target)):raise ValueError("Source-filtered fold index differs")
        old_registry=read(original/"source_registry_v3.json")
        if old_registry!=registry:raise ValueError("Fresh task registry differs")
        arrays=0
        for p in sorted((original/"features").glob("*.npz"))+[original/"planner_eligibility.npz"]:
            target=out/p.relative_to(original);x=dict(np.load(p));y=dict(np.load(target))
            if set(x)!=set(y) or any(not np.array_equal(x[k],y[k],equal_nan=True) for k in x):raise ValueError("Fresh phase/feature/mapping arrays differ")
            arrays+=len(x)
        for p in (original/"native").glob("*.npy"):
            if not np.array_equal(np.load(p),np.load(out/"native"/p.name)):raise ValueError("Independent raw decode differs")
        write(out/"replay.json",dict(status="FRESH_DATA_V3_CHANGE_E1_REPLAY_VERIFIED",optimizer_calls=0,source_raw_decodes=len(sources),
            array_fields_verified=arrays,all_event_ids_replayed=len(events),source_dependencies_replayed=True))
    print("DATA_V3_E1_COMPLETED",len(events),flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ("assets","population","config","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(a.assets,a.population,read(a.config),a.out,a.replay_from)
if __name__=="__main__":main()

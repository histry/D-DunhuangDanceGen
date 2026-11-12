"""Fresh dense continuous AIST observations under Data V3 unseen-song roles."""
import argparse,collections,json,math,subprocess,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write,pinned
from dunhuang.data.sources import ref
from dunhuang.data.conversion import decode
from dunhuang.data.split_registry import checkpoint_binding
from dunhuang.planning.time_map import motion_features74
from dunhuang.alignment.event_targets import music_events,motion_events,crop_events
from dunhuang.alignment.conditional_candidates import nuisance_statistics
from dunhuang.alignment.dense_coverage import real_starts
from dunhuang.research.source_identifiability.a2 import candidates
from .contracts import validate_song_task,Moments,training_gate

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def run(assets_file,config,out,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);(out/"sources").mkdir();(out/"audio").mkdir()
    if config["strict"]!=read("configs/content_identifiability_v2.json")["strict"] or config["test_tensor_reads"]:raise ValueError("Historical challenge thresholds or Test boundary changed")
    assets=read(assets_file);registry=read(config["registry"])
    old_seal=read("reports/data_event_v3_summary.json")["evidence"]["main/source_registry_v3.json"]
    if sha(config["registry"])!=old_seal:raise ValueError("Frozen Data V3 registry differs")
    task=registry["protocols"][config["a2_protocol"]];roles=task["roles"]
    source_info={r["source_sha256"]:r for r in registry["sources"] if r["dataset_id"]=="aistpp"}
    paired=read(checked(assets["paired_generation_manifest"]));selected={}
    for r in paired["rows"]:
        source=source_info.get(r["raw_motion"]["sha256"]);role=roles.get(source["source_uid"]) if source else None
        if role in ("train","val"):selected.setdefault(r["raw_motion"]["sha256"],(r,source,role))
    expected=collections.Counter(role for role in roles.values() if role in ("train","val"))
    if collections.Counter(role for r,source,role in selected.values())!=expected:raise ValueError("New task source intersection incomplete")
    runtime=read(checked(assets["audio_runtime"]))
    for item in runtime["files"]+runtime.get("wheels",[])+[runtime["wheel"]]:checked(item)
    sys.path.insert(0,str(Path(runtime["import_root"]).resolve()))
    legacy=pinned(assets["legacy_root"],assets["legacy_commit"]);sys.path.insert(0,str(legacy))
    import librosa,soxr
    if librosa.__version__!="0.11.0" or soxr.__version__!=runtime["version"]:raise ValueError("Frozen audio environment differs")
    from evaluation.paired_alignment_observations_v3 import full_audio_observations
    norm=dict(music=Moments(50),motion=Moments(74));audio={};rows=[];ms=[];ds=[];mv=[];dv=[];em=[];ed=[];events={};seals=[];source_counts=collections.Counter()
    proof_cache={};checked_assets={}
    def check(r):
        key=(r["path"],r["sha256"])
        if key not in checked_assets:checked_assets[key]=checked(r)
        return checked_assets[key]
    for number,(rk,(r,source,role)) in enumerate(sorted(selected.items())):
        raw=check(r["raw_motion"]);check(r["audio"])
        proof=read(check(r["alignment_attestation"]));calref=proof["source_calibration"]
        if calref["sha256"] not in proof_cache:proof_cache[calref["sha256"]]=read(check(calref))
        cal=proof_cache[calref["sha256"]];c=cal["calibration"]
        if proof["status"]!="verified" or not c["verified"] or not c["origin_anchor_used"] or len(c["anchors"])<3 or not math.isfinite(c["maximum_residual_seconds"]) or c["maximum_residual_seconds"]>1/30:raise ValueError("Unverified continuous source clock")
        if cal["raw_motion"]["sha256"]!=rk or cal["raw_audio"]["sha256"]!=r["audio"]["sha256"] or c["audio_offset_seconds"]!=r["audio_offset_seconds"]:raise ValueError("Clock/source binding differs")
        native,observed=decode(raw,"aistpp");values,mask=motion_features74(native)
        if abs(len(native)-c["motion_seconds"]*30)>1:raise ValueError("Source duration disagrees")
        norm["motion"].add(values,mask,role)
        native_file=out/"sources"/(rk+".npy");np.save(native_file,native);native_ref=ref(native_file)
        de=motion_events(values);ak=r["audio"]["sha256"]
        if ak not in audio:
            av,am,receipt=full_audio_observations(check(r["audio"]));me=music_events(av,am)
            audio[ak]=(av,am,me,role);events[ak]=me;norm["music"].add(av,am,role)
            np.savez_compressed(out/"audio"/(ak+".npz"),values=av,mask=am)
            write(out/"audio"/(ak+".json"),dict(raw_audio=r["audio"],role=role,extractor=receipt,events=me))
            print("A2_V3_AUDIO",len(audio),flush=True)
        av,am,me,audio_role=audio[ak]
        if audio_role!=role:raise ValueError("Audio content crosses new roles")
        starts=real_starts(len(native),len(av),30*r["audio_offset_seconds"],min(c["motion_seconds"],c["decoded_audio_seconds"]),config["stride_frames"])
        for start in starts:
            a=start+15;ma=a+round(30*r["audio_offset_seconds"])
            clean={k:r[k] for k in ("recording_uid","choreography_uid","music_uid","audio_offset_seconds","audio","raw_motion","fps")}
            clean.update(role=role,historical_role=r["role"],source_group=source["music_work_uid"],start_frame=start,frames=150,
                source_native=native_ref,window_uid="v3-dense:"+rk+":"+str(start),source_clock_evidence=calref)
            rows.append(clean)
            for all_events,first,cap,times,valid in ((me,ma,12,ms,mv),(de,a,20,ds,dv)):
                kept=[e for e in crop_events(all_events,first,cap=cap) if e["relative_seconds"]<=118/30+1e-7]
                t=np.arange(cap,dtype=np.float32)/30+4.1;v=np.zeros(cap,bool)
                t[:len(kept)]=[e["relative_seconds"] for e in kept];v[:len(kept)]=True;times.append(t);valid.append(v)
            em.append(av[ma:ma+120,0].copy());ed.append(values[start:start+150,66].copy())
        seals.append(dict(raw_motion=r["raw_motion"],audio=r["audio"],source_clock=calref,role=role,native=native_ref,frames=len(native),windows=len(starts)))
        source_counts[role]+=1
        if number%100==0:print("A2_V3_SOURCES",number+1,len(selected),"WINDOWS",len(rows),flush=True)
    validate_song_task(rows)
    bank=dict(mt=np.stack(ms),dt=np.stack(ds),mv=np.stack(mv),dv=np.stack(dv))
    stats=nuisance_statistics(rows,bank,dict(energy_music=np.stack(em),energy_motion=np.stack(ed)),events)
    dense,evidence=candidates(rows,bank,stats,config["strict"],role_validator=validate_song_task)
    sparse_ids=np.array([i for i,r in enumerate(rows) if r["start_frame"]%config["sparse_stride_frames"]==0],int)
    sparse_rows=[rows[i] for i in sparse_ids];sparse_bank={k:v[sparse_ids] for k,v in bank.items()};sparse_stats={k:v[sparse_ids] for k,v in stats.items()}
    sparse,sparse_evidence=candidates(sparse_rows,sparse_bank,sparse_stats,config["strict"],role_validator=validate_song_task)
    train_ids=sorted(uid for uid,role in roles.items() if role=="train")
    binding=checkpoint_binding(registry,config["a2_protocol"],train_ids)
    norm_payload=dict(schema="v3_task_statistics_v1",task=config["a2_protocol"],registry_identity=registry["identity_digest"],
        statistics={k:v.result() for k,v in norm.items()},population_binding=binding,task_checkpoint_built=False,
        old_encoder_statistics_replaced=False,test_tensor_reads=0)
    write(out/"train_statistics.json",norm_payload)
    gate=training_gate(evidence["populations"],task,checkpoint_binding=None)
    protocol=dict(schema="a2_v3_support_audit",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config=config,assets=ref(assets_file),registry=ref(config["registry"]),paired_manifest=assets["paired_generation_manifest"],
        statistics=ref(out/"train_statistics.json"),population_binding=binding,global_checkpoint=None,
        optimizer_calls=0,encoder_forward_calls=0,test_tensor_reads=0,new_blind_source=False)
    write(out/"protocol.json",protocol);write(out/"manifest.json",dict(rows=rows,test_tensor_reads=0))
    write(out/"source_seals.json",seals);write(out/"candidate_evidence.json",evidence)
    np.savez_compressed(out/"nodes.npz",**bank);np.savez_compressed(out/"nuisance.npz",**stats)
    np.savez_compressed(out/"candidates.npz",**dense);np.savez_compressed(out/"sparse_candidates.npz",**sparse,sparse_ids=sparse_ids)
    result=dict(schema="a2_v3_qualification_result",sources=dict(source_counts),audio_assets=len(audio),windows=len(rows),sparse_windows=len(sparse_rows),
        dense_population=evidence["populations"],sparse_population=sparse_evidence["populations"],
        dense_rule_failure_intersections=evidence["rule_failure_intersections"],sparse_rule_failure_intersections=sparse_evidence["rule_failure_intersections"],
        gate=gate,training_status="NOT_STARTED_UNQUALIFIED_NEW_CHECKPOINT_LINEAGE",optimizer_calls=0,
        statistics_fit_calls=2,encoder_forward_calls=0,test_tensor_reads=0,old_thresholds_unchanged=True,
        role_closure="music/audio/recording/raw SHA; choreography may be shared for unseen-song task",
        old_encoder_norm_reused=False,work_aliases_certified=False,pretrained_exposure_certified=False,operational_qualified=False)
    write(out/"result.json",result)
    if replay_from:
        original=Path(replay_from)
        for name in ("result.json","train_statistics.json","candidate_evidence.json"):
            if read(original/name)!=read(out/name):raise ValueError("A2 V3 fresh scientific replay differs: "+name)
        fields=0
        for name in ("nodes.npz","nuisance.npz","candidates.npz","sparse_candidates.npz"):
            x=dict(np.load(original/name));y=dict(np.load(out/name))
            if set(x)!=set(y) or any(not np.array_equal(x[k],y[k]) for k in x):raise ValueError("A2 V3 fresh arrays differ")
            fields+=len(x)
        for file in (original/"sources").glob("*.npy"):
            if not np.array_equal(np.load(file),np.load(out/"sources"/file.name)):raise ValueError("A2 V3 raw source decode differs")
        for file in (original/"audio").glob("*.npz"):
            x=dict(np.load(file));y=dict(np.load(out/"audio"/file.name))
            if any(not np.array_equal(x[k],y[k]) for k in x):raise ValueError("A2 V3 raw waveform observations differ")
        write(out/"replay.json",dict(status="FRESH_V3_RAW_AUDIO_MOTION_CANDIDATE_REPLAY",optimizer_calls=0,source_decodes=len(selected),audio_reextractions=len(audio),array_fields_verified=fields,test_tensor_reads=0))
    print("A2_V3_COMPLETED",json.dumps(evidence["populations"]),flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ("assets","config","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(a.assets,read(a.config),a.out,a.replay_from)
if __name__=="__main__":main()

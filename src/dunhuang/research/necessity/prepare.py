"""Qualify G2-M on exact self targets and independently captured same-clock pairs."""
import argparse,collections,copy,json,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.data.conversion import decode
from dunhuang.data.split_registry import stable
from dunhuang.data.change_event_phases import motion_profiles,detect
from dunhuang.research.transfer_audit.contracts import validate_song_task
from dunhuang.generation.targets import phase_target
from dunhuang.generation.source_pairs import source_pairs
from dunhuang.research.source_identifiability.g2 import make_condition
from dunhuang.alignment.audio_clock import music_seconds
from dunhuang.alignment.sampling import group_schedule,coverage
from .factors import phase_mask,donors,timesteps,visible_source_distance

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def run(assets_file,config,out,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);assets=read(assets_file);cfg=config["g2"]
    registry=read(config["registry"]);roles=registry["protocols"][config["task"]]["roles"]
    if sha(config["registry"])!=read("reports/data_event_v3_summary.json")["evidence"]["main/source_registry_v3.json"]:raise ValueError("Frozen registry differs")
    source_info={s["source_sha256"]:s for s in registry["sources"] if s["dataset_id"]=="aistpp"}
    paired=read(checked(assets["paired_generation_manifest"]));rows=[]
    for r in paired["rows"]:
        s=source_info[r["raw_motion"]["sha256"]];role=roles.get(s["source_uid"])
        if role in ("train","val"):rows.append(dict(r,role=role,historical_role=r["role"],source_group=s["music_work_uid"]))
    rows=sorted(rows,key=lambda r:r["window_uid"]);validate_song_task(rows)
    cache={};audio_cache={};phase_cache={};source_seals=[];masks={};admitted=[];statuses=collections.Counter()
    def check(r):
        k=(r["path"],r["sha256"])
        if k not in cache:cache[k]=checked(r)
        return cache[k]
    with np.load(check(assets["scaler"])) as z:scale,bias=z["scale"],z["bias"]
    bridge=out/"audio_contract";(bridge/"data/sidecars").mkdir(parents=True);bridge_files={}
    raw_root=out/"sources";raw_root.mkdir()
    for i,r in enumerate(rows):
        check(r["raw_motion"]);check(r["audio"]);proof=read(check(r["alignment_attestation"]))
        if proof["status"]!="verified" or proof["motion_sha256"]!=r["raw_motion"]["sha256"] or proof["audio_sha256"]!=r["audio"]["sha256"] or proof["audio_offset_seconds"]!=r["audio_offset_seconds"]:raise ValueError("Verified clock differs")
        native=np.load(check(r["motion"]));observed=np.load(check(r["observed"]));supervised=np.load(check(r["supervision"]))
        if native.shape!=(150,151) or observed.dtype!=bool or not observed[:,79:127].all() or not supervised.all():raise ValueError("Real full target observations required")
        if phase_target(native,scale,bias)[0] is None:raise ValueError("Frozen Scaler support differs")
        receipt=read(check(r["jukebox_receipt"]));juke=np.load(check(r["jukebox"]),mmap_mode="r")
        if juke.shape!=(150,4800) or juke.dtype!=np.float32 or not np.isfinite(juke).all():raise ValueError("Real Jukebox tensor differs")
        if receipt["features_sha256"]!=r["jukebox"]["sha256"] or receipt["window_uid"]!=r["window_uid"] or receipt["audio_sha256"]!=r["audio"]["sha256"] or receipt["layer"]!=66 or receipt["fps"]!=30 or receipt["silence_padding"]:raise ValueError("Jukebox receipt differs")
        if abs(receipt["sample_start"]/receipt["sample_rate"]-float(music_seconds(r)))>1/receipt["sample_rate"]:raise ValueError("Jukebox physical clock differs")
        ak=r["audio"]["sha256"];rk=r["raw_motion"]["sha256"]
        if ak not in audio_cache:
            a=Path(config["source_a2"])/"audio"/(ak+".json");b=Path(config["source_a2_replay"])/"audio"/(ak+".json")
            if read(a)!=read(b):raise ValueError("Frozen raw waveform event replay differs")
            payload=read(a)
            if payload["raw_audio"]["sha256"]!=ak:raise ValueError("Audio event source binding differs")
            sidecar=bridge/"data/sidecars"/(ak+".json");write(sidecar,payload)
            bridge_files["data/sidecars/"+sidecar.name]=sha(sidecar);audio_cache[ak]=ref(a)
            write(bridge/"preflight.json",dict(schema="real_waveform_sidecar_bridge_v1",files=bridge_files))
        if rk not in phase_cache:
            full,_=decode(check(r["raw_motion"]),"aistpp")
            profile=motion_profiles(full,list(range(24)));phases=detect(profile,cfg["phase_config"])
            file=raw_root/(rk+".npy");np.save(file,full);phase_cache[rk]=(full,phases)
            source_seals.append(dict(raw_motion=r["raw_motion"],fresh_native=ref(file),frames=len(full),derivative_support=[0,len(full)]))
        full,phases=phase_cache[rk];a=r["start_frame"]
        if not np.array_equal(full[a:a+150,7:],native[:,7:]):raise ValueError("Self target does not match raw recording pose")
        condition,keys,_,_=make_condition(r,bridge)
        known=condition.known[0,:,79:127].reshape(150,8,6).all(-1).numpy();original=keys[0].numpy()
        expanded=phase_mask(phases,a,known,original,cfg["phase_radius"])
        masks[r["window_uid"]]=dict(original=original,expanded=expanded,known=known)
        status=int(expanded.any(-1).sum())>=cfg["min_phase_frames"]
        statuses[(r["role"],"phase_admitted" if status else "insufficient_phase")]+=1
        if status:admitted.append(r)
        if i%300==0:print("G2M_PREFLIGHT",i,len(rows),flush=True)
    index={r["window_uid"]:r for r in admitted};pairs=source_pairs(dict(rows=admitted))
    unique={};comparison_counts=collections.Counter()
    for p in pairs:
        a,b=p["target_uid"],p["source_uid"];ra,rb=index[a],index[b]
        key=tuple(sorted((a,b)))
        if key in unique:continue
        na=np.load(check(ra["motion"]));nb=np.load(check(rb["motion"]))
        distance=visible_source_distance(na,nb,masks[a]["known"],masks[b]["known"])
        if distance<cfg["min_reference_separation_rad"]:comparison_counts[(ra["role"],"below_source_separation")]+=1;continue
        if not np.array_equal(np.load(check(ra["jukebox"])),np.load(check(rb["jukebox"]))):comparison_counts[(ra["role"],"music_tensor_differs")]+=1;continue
        unique[key]=dict(a=min(a,b),b=max(a,b),role=ra["role"],music_uid=ra["music_uid"],separation_rad=distance)
    valid_pairs=list(unique.values());val_pairs=[p for p in valid_pairs if p["role"]=="val"]
    by_music={}
    for p in sorted(val_pairs,key=lambda p:stable([config["seed"],p["a"],p["b"]])):by_music.setdefault(p["music_uid"],p)
    chosen=[by_music[k] for k in sorted(by_music,key=lambda k:stable([config["seed"],k]))[:cfg["music_cases"]]]
    train=[r for r in admitted if r["role"]=="train"]
    train_records=len({r["recording_uid"] for r in train});train_music=len({r["music_uid"] for r in train})
    val_records=len({index[uid]["recording_uid"] for p in val_pairs for uid in (p["a"],p["b"])})
    checks=dict(real_targets_clocks_jukebox_verified=True,train_windows=len(train)>=cfg["min_train_windows"],
        train_recordings=train_records>=cfg["min_train_recordings"],train_music=train_music>=cfg["min_train_music"],
        val_music=len(by_music)>=cfg["min_val_music"],val_recordings=val_records>=cfg["min_val_recordings"],
        all_registered_cases=len(chosen)==cfg["music_cases"],test_tensor_reads_zero=True,known_role_closure_verified=True)
    gate=dict(qualified=all(checks.values()),checks=checks,scope=cfg["scope"],
        aliases_outside_publisher_ids="UNKNOWN",pretrained_exposure="UNKNOWN",blind_generalization_qualified=False,
        generic_data_v3_training_flag_promoted=False,operational_qualified=False)
    schedule=group_schedule(train,cfg["steps"],cfg["batch"],config["seed"])
    corruption=donors(schedule,train,config["seed"]+31)
    u=np.random.default_rng(config["seed"]+101).random(schedule.shape)
    np.savez_compressed(out/"schedule.npz",rows=schedule,donors=corruption,uniform=u,old_t=timesteps(u,False),ddim_t=timesteps(u,True))
    np.savez_compressed(out/"supervision.npz",**{uid+"__"+kind:v for uid,m in masks.items() for kind,v in m.items()})
    old_protocol=Path("outputs/source_id_03/g2/protocol.json")
    if sha(old_protocol)!=read("outputs/source_id_03/public_summary.json")["evidence"]["g2/protocol.json"]:raise ValueError("Frozen initialization protocol differs")
    initial=read(old_protocol)["initial_adapter"];check(initial)
    manifest=dict(schema="g2m_exact_self_target_tasks",rows=admitted,train_uids=[r["window_uid"] for r in train],
        validation_pairs=chosen,audio_contract=str(bridge),initial_adapter=initial,paired_manifest=assets["paired_generation_manifest"])
    write(out/"manifest.json",manifest);write(out/"source_seals.json",source_seals)
    binding=stable(dict(task=cfg["scope"],registry=registry["identity_digest"],train=[(r["window_uid"],r["raw_motion"]["sha256"]) for r in train],
        source_masks=sha(out/"supervision.npz"),schedule=sha(out/"schedule.npz"),initial=initial["sha256"],scaler=assets["scaler"]["sha256"],base=assets["checkpoint"]["sha256"]))
    protocol=dict(schema="g2m_preflight_v1",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config=config,assets=ref(assets_file),registry=ref(config["registry"]),task_binding=binding,
        manifest=ref(out/"manifest.json"),supervision=ref(out/"supervision.npz"),schedule=ref(out/"schedule.npz"),
        gate=gate,test_tensor_reads=0,checkpoint_selection="fixed_last_no_validation_selection")
    write(out/"protocol.json",protocol)
    report=dict(qualification=gate,admitted_windows=dict(collections.Counter(r["role"] for r in admitted)),train_recordings=train_records,train_music=train_music,
        val_pair_groups=len(val_pairs),val_music=len(by_music),val_recordings=val_records,registered_cases=len(chosen),
        phase_status={str(k):v for k,v in statuses.items()},pair_rejections={str(k):v for k,v in comparison_counts.items()},
        source_decodes=len(phase_cache),real_audio_event_receipts=len(audio_cache),schedule_coverage=coverage(train,schedule),
        original_key_frames=dict(collections.Counter(int(masks[r["window_uid"]]["original"].any(-1).sum()) for r in admitted)),
        expanded_phase_frames=dict(collections.Counter(int(masks[r["window_uid"]]["expanded"].any(-1).sum()) for r in admitted)),
        optimizer_calls=0,formal_data_v3_training_qualified=False,operational_qualified=False)
    write(out/"result.json",report)
    if replay_from:
        old=Path(replay_from)
        if read(old/"result.json")!=read(out/"result.json"):raise ValueError("G2M source qualification replay differs")
        for name in ("schedule.npz","supervision.npz"):
            a=dict(np.load(old/name));b=dict(np.load(out/name))
            if set(a)!=set(b) or any(not np.array_equal(a[k],b[k]) for k in a):raise ValueError("G2M supervision/schedule replay differs")
        if read(old/"protocol.json")["task_binding"]!=binding:raise ValueError("G2M task lineage differs")
        write(out/"replay.json",dict(status="FRESH_G2M_SOURCE_TASK_REPLAY",optimizer_calls=0,source_decodes=len(phase_cache)))
    print("G2M_QUALIFIED",gate["qualified"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("assets","config","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(a.assets,read(a.config),a.out,a.replay_from)

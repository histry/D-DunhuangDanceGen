"""Hash-sealed event observations from full sources and previously verified clocks."""
import json,pickle,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,sha,write,pinned
from dunhuang.data.conversion import decode
from dunhuang.planning.time_map import motion_features74
from .clock_data import read,ref
from .event_targets import EXTRACTOR,motion_events,music_events,crop_events,review_packet
from .event_metrics import source_clock_masks
from .sampling import coverage

def interpolate(tokens,seconds):
    p=np.asarray(seconds)*15
    if len(p) and (p.min()<-1e-6 or p.max()>len(tokens)-1+1e-6): raise ValueError("Event outside actual token support")
    lo=np.floor(p).astype(int);hi=np.ceil(p).astype(int);alpha=(p-lo)[:,None]
    return (tokens[lo]*(1-alpha)+tokens[hi]*alpha).astype(np.float32)

def event_bank(events,tokens,cap,descriptors):
    events=[e for e in events if e["relative_seconds"]<=118/30+1e-7]
    out=np.zeros((cap,144),np.float32);valid=np.zeros(cap,bool);times=np.arange(cap,dtype=np.float32)/30+4.1
    if events:
        n=len(events);t=[e["relative_seconds"] for e in events]
        out[:n,:128]=interpolate(tokens,t);out[:n,128:]=np.stack([descriptors(e) for e in events]);valid[:n]=True;times[:n]=t
    return out,valid,times,events

def candidates(rows,arrays,eligible,seed,count=8):
    # Same song / correct source time has multiple legitimate capture positives.
    rng=np.random.default_rng(seed); result=np.full((len(rows),count),-1,np.int64);pos=np.zeros_like(result,bool);valid=pos.copy()
    mm=arrays["energy_music"][:,::2];dd=arrays["energy_motion"][:,15:135:2]
    def unit(x):
        x=x-x.mean(-1,keepdims=True);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
    mm=unit(mm);dd=unit(dd)
    for i,r in enumerate(rows):
        if not eligible[i]:continue
        pool=[j for j,x in enumerate(rows) if x["role"]==r["role"] and eligible[j]]
        p,v=source_clock_masks(r,[rows[j] for j in pool]);positives=[pool[k] for k in np.flatnonzero(p)]
        # Own capture first plus another legitimate interpretation when available.
        positives=[i]+[j for j in positives if j!=i];positives=positives[:2]
        same=[j for j in pool if rows[j]["music_uid"]==r["music_uid"] and abs(rows[j]["start_frame"]-r["start_frame"])>=120]
        unrelated=[pool[k] for k in np.flatnonzero(v&~p) if rows[pool[k]]["music_uid"]!=r["music_uid"]]
        if len(same)<2 or len(unrelated)<count-len(positives)-2:continue
        same=list(rng.permutation(same))[:2]
        scores=dd[unrelated]@mm[i];order=np.argsort(-scores,kind="stable")
        energy=[unrelated[k] for k in order[:count-len(positives)-len(same)]]
        chosen=positives+same+energy
        result[i]=chosen;p,v=source_clock_masks(r,[rows[j] for j in chosen]);pos[i]=p;valid[i]=v
    return result,pos,valid

def prepare(source,out,config,assets):
    source=Path(source).resolve();out=Path(out);out.mkdir();(out/"sidecars").mkdir();(out/"annotation_review").mkdir()
    manifest=read(source/"data/manifest.json");rows=manifest["rows"]
    if read(source/"result.json")["status"]!="COMPLETED_CLOCK_ALIGNMENT_SHADOW" or manifest["test_rows_extracted"]!=0: raise ValueError("Verified clock source required")
    roles={}
    for r in rows:
        if r["role"] not in ("train","val"): raise ValueError("Test extraction forbidden")
        for key in ("source_group","recording_uid","music_uid","choreography_uid"):
            identity=(key,r[key])
            if identity in roles and roles[identity]!=r["role"]: raise ValueError("Associated identity crosses roles")
            roles[identity]=r["role"]
        checked(r["alignment_attestation"])
    legacy=pinned(assets["legacy_root"],assets["legacy_commit"]);sys.path.insert(0,str(legacy))
    from evaluation.paired_alignment_observations_v3 import motion_from_body_rotations
    from dunhuang.evaluation_benchmarks.native_shadow import LegacyBackend
    backend=LegacyBackend(assets)
    enc=dict(np.load(source/"encodings.npz"));obs=dict(np.load(source/"data/observations.npz"))
    norm=manifest["norm"]["motion"]
    raw=np.stack([np.load(source/"data/windows"/("%05d.npz"%i))["motion"] for i in range(len(rows))])
    old=np.empty_like(raw);fk_features=np.empty((len(rows),120,128),np.float32)
    target=np.empty((len(rows),120,3),np.float32)
    music_cache={};motion_cache={};seals={}
    banks={k:[] for k in ("music","motion","mv","dv","mt","dt")}
    selected_events=[]; packet_groups=set()
    for i,r in enumerate(rows):
        for key in ("audio","raw_motion"):
            rr=r[key]
            if rr["sha256"] not in seals: checked(rr);seals[rr["sha256"]]=rr
        ak=r["audio"]["sha256"];rk=r["raw_motion"]["sha256"]
        if ak not in music_cache:
            q=np.load(source/"data/audio"/(ak+".npz"));events=music_events(q["values"],q["mask"]);music_cache[ak]=(events,q["values"].copy())
            write(out/"sidecars"/(ak+".json"),dict(source=r["audio"],extractor=EXTRACTOR,events=events))
        if rk not in motion_cache:
            native,_=decode(checked(r["raw_motion"]),r["dataset"]);features,mask=motion_features74(native)
            with checked(r["raw_motion"]).open("rb") as f:original=pickle.load(f)
            pose=np.asarray(original["smpl_poses"],np.float32).reshape(-1,24,3)[:,:22]
            rotations=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
            legacy_values,_,_=motion_from_body_rotations(rotations,native_fps=60.)
            fk=backend.fk(native)
            velocity=np.empty_like(fk);velocity[1:]=np.diff(fk,axis=0)*30;velocity[0]=velocity[1]
            # Independently computed Cartesian kinematics, not force/contact/semantic annotations.
            upper=np.linalg.norm(velocity[:,(12,13,14,15,16,17,18,19,20,21)],axis=-1).mean(-1)
            lower=np.linalg.norm(velocity[:,(1,2,4,5,7,8,10,11)],axis=-1).mean(-1)
            root=np.linalg.norm(velocity[:,0],axis=-1)
            targets=np.stack((upper,lower,root),-1).astype(np.float32)
            ff=np.pad(np.concatenate((velocity.reshape(-1,72),(fk[:,:18]-fk[:,:1]).reshape(-1,54)),-1),((0,0),(0,2))).astype(np.float32)
            events=motion_events(features)
            from .event_targets import peaks
            cartesian={}
            for column,part in enumerate(("upper","lower","root")):
                velocity_signal=targets[:,column]
                cartesian[(part,"speed_peak")]=peaks(velocity_signal,"speed_peak",part)
                cartesian[(part,"start")]=peaks(np.maximum(np.diff(velocity_signal,prepend=velocity_signal[0])*30,0),"start",part)
                cartesian[(part,"deceleration")]=peaks(np.maximum(-np.diff(velocity_signal,prepend=velocity_signal[0])*30,0),"deceleration",part)
            for e in events:
                e["detector_agreement"]=any(abs(e["frame"]-q["frame"])<=3 for q in cartesian.get((e["body_part"],e["event_type"]),[]))
            motion_cache[rk]=(features,legacy_values,ff,targets,events)
            write(out/"sidecars"/(rk+".json"),dict(source=r["raw_motion"],extractor=EXTRACTOR,
                skeleton_sha256=backend.skeleton_hash,units="angular rad/s; FK Cartesian m/s",events=events))
        features,previous,ff,targets,events=motion_cache[rk]
        start=r["start_frame"];a=start+15;b=start+135
        if start+150>min(len(features),len(previous)): raise ValueError("Shared continuous operator support unavailable")
        if not np.array_equal(raw[i],features[start:start+150]): raise ValueError("Continuous source differs from frozen cache")
        old[i]=previous[start:start+150];fk_features[i]=ff[a:b];target[i]=targets[a:b]
        me=crop_events(music_cache[ak][0],a+30*r["audio_offset_seconds"],cap=config["music_event_cap"])
        de=crop_events(events,a,cap=config["motion_event_cap"])
        def music_descriptor(e):
            v=np.zeros(16,np.float32);frame=e["frame"];audio_values=music_cache[ak][1]
            v[:12]=audio_values[frame,18:30];v[12+("onset","accent","beat","silence").index(e["event_type"])]=1;return v
        def motion_descriptor(e):
            from .event_targets import PARTS,TYPES
            v=np.zeros(16,np.float32);vector=features[e["frame"],:66].reshape(22,3)[list(PARTS[e["body_part"]])]
            v[:3]=vector.mean(0);speeds=np.linalg.norm(vector,axis=-1);v[3]=speeds.mean();v[4]=speeds.max()
            v[5+TYPES.index(e["event_type"])]=1;v[10+list(PARTS).index(e["body_part"])]=1;v[13]=e["confidence"];return v
        m,mv,mt,me=event_bank(me,enc["mt"][i],config["music_event_cap"],music_descriptor)
        d,dv,dt,de=event_bank(de,enc["dt"][i,2],config["motion_event_cap"],motion_descriptor)
        for k,v in dict(music=m,motion=d,mv=mv,dv=dv,mt=mt,dt=dt).items():banks[k].append(v)
        selected_events.append(dict(window_uid=r["window_uid"],music=me,motion=de))
        group=(r["role"],r["source_group"])
        if group not in packet_groups:
            write(out/"annotation_review"/("%05d.json"%i),review_packet(r));packet_groups.add(group)
        if (i+1)%200==0:print("EVENT_OBSERVATIONS",i+1,len(rows),flush=True)
    bank={k:np.stack(v) for k,v in banks.items()}
    descriptor_norm={}
    train=np.array([r["role"]=="train" for r in rows])
    for name,mask in (("music","mv"),("motion","dv")):
        values=bank[name][train,:,128:][bank[mask][train]]
        mean=values.mean(0);std=values.std(0);std=np.where(std<1e-6,1.,std)
        bank[name][:,:,128:]=(bank[name][:,:,128:]-mean)/std
        bank[name]*=bank[mask][...,None]
        descriptor_norm[name]=dict(mean=mean.tolist(),std=std.tolist())
    eligible=bank["mv"].any(-1)&bank["dv"].any(-1)
    cand,pos,valid=candidates(rows,obs,eligible,config["seed"],config["candidates"])
    eligible=eligible&(cand>=0).all(-1)&pos.any(-1)&(valid&~pos).any(-1)
    from .event_weak_labels import weak_labels
    weak_target,weak_weight,weak_evidence=weak_labels(rows,selected_events,config["band_seconds"])
    wt=np.zeros((len(rows),config["music_event_cap"],config["motion_event_cap"]),np.float32)
    ww=np.zeros((len(rows),config["music_event_cap"]),np.float32)
    wt[:,:weak_target.shape[1],:weak_target.shape[2]]=weak_target;ww[:,:weak_weight.shape[1]]=weak_weight
    write(out/"weak_evidence.json",weak_evidence)
    np.savez_compressed(out/"events.npz",weak_target=wt,weak_weight=ww,**bank,candidates=cand,positive=pos,valid=valid,eligible=eligible)
    # Equal 128-input linear probes: raw74 uses explicit structural zeros; these are NOT motion truth.
    raw_probe=np.pad(obs["motion"][:,15:135],((0,0),(0,0),(0,54)))
    np.savez_compressed(out/"probes.npz",raw74=raw_probe,tcn=enc["tcn"],transformer=enc["dt"][:,2],fk=fk_features,target=target)
    np.savez_compressed(out/"operator.npz",old=old,current=raw,mask=obs["dm"])
    write(out/"selected_events.json",selected_events);write(out/"source_seals.json",list(seals.values()))
    result=dict(schema="event_study_data_v1",rows=rows,source_manifest=ref(source/"data/manifest.json"),
        source_encodings=ref(source/"encodings.npz"),source_observations=ref(source/"data/observations.npz"),
        source_training_schedule=ref(source/"training_schedule.npy"),global_checkpoint=manifest["global_checkpoint"],norm=manifest["norm"],
        old_schedule_coverage=coverage(rows,np.load(source/"training_schedule.npy")),
        extractor=EXTRACTOR,event_node_width=144,descriptor_normalization=descriptor_norm,skeleton_sha256=backend.skeleton_hash,eligible_rows=int(eligible.sum()),
        train_eligible=int(sum(eligible[i] for i,r in enumerate(rows) if r["role"]=="train")),
        val_eligible=int(sum(eligible[i] for i,r in enumerate(rows) if r["role"]=="val")),
        observations_are_independent_correspondence_labels=False,test_rows_extracted=0,
        weak_label_states=weak_evidence["states"],annotation_status="UNREVIEWED",event_interval="real 120-frame core; max event time118/30 to stay inside actual 15Hz token support")
    if not result["train_eligible"] or not result["val_eligible"]:raise ValueError("Empty fixed candidate population")
    write(out/"manifest.json",result)
    return result

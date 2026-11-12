"""A1: fresh dense real windows with whole-record clock receipts; no new pose truth."""
import json,math
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write,sha
from dunhuang.data.conversion import decode
from dunhuang.planning.time_map import motion_features74
from .clock_data import read,ref
from .event_targets import crop_events,PARTS,TYPES
from .conditional_candidates import nuisance_statistics,match_statistics,validate_roles
from .model import RetrievalModel

def real_starts(native_frames,audio_frames,offset_frames,verified_seconds,stride):
    if type(stride) is not int or stride<=0:raise ValueError("Positive registered stride required")
    if abs(offset_frames-round(offset_frames))>1e-6:raise ValueError("Exact integer source/audio grid required")
    support=min(native_frames,int(np.floor(verified_seconds*30+1e-6))+1)
    return [s for s in range(0,max(0,support-150+1),stride)
        if s+round(offset_frames)>=0 and s+round(offset_frames)+149<audio_frames]

def sample_tokens(tokens,times,fps):
    p=np.asarray(times)*fps
    if len(p) and (p.min()<0 or p.max()>len(tokens)-1+1e-6):raise ValueError("Unobserved event time; no extrapolation")
    lo=np.floor(p).astype(int);hi=np.ceil(p).astype(int);a=(p-lo)[:,None]
    return (tokens[lo]*(1-a)+tokens[hi]*a).astype(np.float32)

def fast_candidates(rows,bank,stats,config):
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r["role"],r["music_uid"],r["choreography_uid"]),[]).append(i)
    n=len(rows);c=np.repeat(np.arange(n)[:,None],4,axis=1);positive=np.zeros((n,4),bool);valid=positive.copy();eligible=np.zeros(n,bool)
    evidence=[];failures={k:0 for k in ("count","time_distribution","histogram","energy","onset")};memo={}
    for i,r in enumerate(rows):
        if not bank["mv"][i].any() or not bank["dv"][i].any():continue
        pool=groups[(r["role"],r["music_uid"],r["choreography_uid"])]
        pos=[i]+[j for j in pool if j!=i and rows[j]["start_frame"]==r["start_frame"] and bank["dv"][j].any()]
        negatives=[]
        for j in pool:
            if abs(rows[j]["start_frame"]-r["start_frame"])<120 or not bank["mv"][j].any() or not bank["dv"][j].any():continue
            key=(min(i,j),max(i,j))
            if key not in memo:memo[key]=match_statistics(stats,i,j,config)
            ok,detail=memo[key]
            for name,value in detail["rules"].items():failures[name]+=not value
            evidence.append(dict(query=i,candidate=j,accepted=ok,**detail))
            if ok:negatives.append((detail["selection_distance"],j))
        if negatives:
            chosen=pos[:2]+[j for _,j in sorted(negatives)[:2]]
            c[i,:len(chosen)]=chosen;positive[i,:min(2,len(pos))]=True;valid[i,:len(chosen)]=True;eligible[i]=True
        if (i+1)%1000==0:print("DENSE_CANDIDATES",i+1,n,flush=True)
    populations={}
    for role in ("train","val"):
        ids=[i for i,r in enumerate(rows) if r["role"]==role and eligible[i]]
        components=len({rows[i]["source_group"] for i in ids});records=len({rows[i]["recording_uid"] for i in ids})
        populations[role]=dict(queries=len(ids),recordings=records,components=components,population_sufficient=
            len(ids)>=config["min_queries"] and records>=config["min_recordings"] and components>=config["min_"+role+"_components"])
    return dict(candidates=c,positive=positive,valid=valid,eligible=eligible),dict(populations=populations,
        failed_rule_counts=failures,comparison_count=len(evidence),evidence=evidence,rules_relaxed=False)

def build(source,clock,out,config):
    source=Path(source);clock=Path(clock);out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/"sources").mkdir()
    parent=read(source/"data/manifest.json");parent_sha=sha(source/"data/manifest.json");old=parent["rows"];validate_roles(old)
    for name,digest in read(source/"preflight.json")["files"].items():
        if name.startswith("data/sidecars/") and sha(source/name)!=digest:raise ValueError("Full source event cache seal differs")
    original=read(clock/"data/manifest.json")
    if sha(clock/"encodings.npz")!=parent["source_encodings"]["sha256"]:raise ValueError("Frozen population encoding seal differs")
    global_ref=parent["global_checkpoint"];checked(global_ref)
    sources={};audio={};calibration_refs={};sidecars={};seals={}
    for r in old:
        e=read(checked(r["alignment_attestation"]))
        if e["status"]!="verified" or e["window_uid"]!=r["window_uid"] or e["motion_sha256"]!=r["raw_motion"]["sha256"] or e["audio_sha256"]!=r["audio"]["sha256"]:raise ValueError("Parent source clock receipt differs")
        cr=e["source_calibration"];calibration=read(checked(cr));c=calibration["calibration"]
        if not c["verified"] or not c["origin_anchor_used"] or len(c["anchors"])<3 or not math.isfinite(c["maximum_residual_seconds"]) or c["maximum_residual_seconds"]>1/30:
            raise ValueError("Whole-record clock support not verified")
        if calibration["raw_motion"]["sha256"]!=r["raw_motion"]["sha256"] or calibration["raw_audio"]["sha256"]!=r["audio"]["sha256"] or abs(c["audio_offset_seconds"]-r["audio_offset_seconds"])>1e-8:
            raise ValueError("Full-record clock source binding differs")
        rk=r["raw_motion"]["sha256"];ak=r["audio"]["sha256"]
        if rk not in sources:
            checked(r["raw_motion"]);native,_=decode(checked(r["raw_motion"]),r["dataset"])
            if abs(c["motion_seconds"]*30-len(native))>1:raise ValueError("Raw recording duration differs from clock calibration")
            values,mask=motion_features74(native);file=out/"sources"/(rk+".npy");np.save(file,native)
            events=read(source/"data/sidecars"/(rk+".json"))["events"]
            sources[rk]=(r,native,values,mask,events,ref(file),min(c["motion_seconds"],c["decoded_audio_seconds"]))
            calibration_refs[rk]=cr;seals[rk]=r["raw_motion"]
        elif any(sources[rk][0][k]!=r[k] for k in ("role","recording_uid","music_uid","choreography_uid","source_group","audio_offset_seconds")):
            raise ValueError("Identical raw source has conflicting identities")
        if ak not in audio:
            checked(r["audio"]);z=np.load(clock/"data/audio"/(ak+".npz"))
            seals["cache:"+ak]=ref(clock/"data/audio"/(ak+".npz"));audio[ak]=(z["values"],z["mask"]);sidecars[ak]=read(source/"data/sidecars"/(ak+".json"))["events"];seals[ak]=r["audio"]
    rows=[];features={k:[] for k in ("music","motion","mm","dm","energy_music","energy_motion")};selected=[];source_sizes=[]
    for rk in sorted(sources):
        r,native,motion,dm,events,native_ref,support_seconds=sources[rk];ak=r["audio"]["sha256"];values,mm=audio[ak]
        starts=real_starts(len(native),len(values),30*r["audio_offset_seconds"],support_seconds,config["stride_frames"]);source_sizes.append(len(starts))
        for start in starts:
            a=start+15;ma=a+round(30*r["audio_offset_seconds"])
            clean={k:r[k] for k in ("dataset","role","source_group","recording_uid","choreography_uid","music_uid","audio_offset_seconds","audio","raw_motion")}
            clean.update(window_uid="dense:"+rk+":"+str(start),start_frame=start,frames=150,fps=30,
                source_native=native_ref,source_clock_evidence=calibration_refs[rk],source_support_frames=len(native),clock_supported_source_frames=min(len(native),int(np.floor(support_seconds*30+1e-6))+1),
                label_scope="verified source clock only; no new Jukebox or 151D generation target",parent_population_sha256=parent_sha)
            rows.append(clean)
            arrays=dict(music=values[ma:ma+120],mm=mm[ma:ma+120],motion=motion[a:a+120],dm=dm[a:a+120],
                energy_music=values[ma:ma+120,0],energy_motion=motion[a:a+120,66])
            for k,v in arrays.items():features[k].append(v.copy())
            me=[e for e in crop_events(sidecars[ak],ma,cap=12) if e["relative_seconds"]<=118/30+1e-7]
            de=[e for e in crop_events(events,a,cap=20) if e["relative_seconds"]<=118/30+1e-7]
            selected.append(dict(music=me,motion=de))
        if len(source_sizes)%100==0:print("DENSE_SOURCES",len(source_sizes),len(sources),"WINDOWS",len(rows),flush=True)
    validate_roles(rows);arrays={k:np.stack(v) for k,v in features.items()}
    norm=parent["norm"]
    for name,mask in (("music","mm"),("motion","dm")):arrays[name]=(((arrays[name]-np.asarray(norm[name]["mean"]))/np.asarray(norm[name]["std"]))*arrays[mask]).astype(np.float32)
    model=RetrievalModel().cuda().eval();model.load_state_dict(torch.load(checked(global_ref),map_location="cuda",weights_only=True)["state"])
    bank={k:[] for k in ("music_tcn30","music_tcn15","music_transformer15","motion","mv","dv","mt","dt")}
    dn=parent["descriptor_normalization"]
    for start in range(0,len(rows),32):
        ids=list(range(start,min(start+32,len(rows))))
        with torch.no_grad():
            m=torch.as_tensor(arrays["music"][ids],device="cuda");d=torch.as_tensor(arrays["motion"][ids],device="cuda")
            m_mask=torch.as_tensor(arrays["mm"][ids],device="cuda");d_mask=torch.as_tensor(arrays["dm"][ids],device="cuda")
            tcn,_=model.music.temporal_features(m,m_mask);_,mt,_=model.music(m,m_mask);_,dt,_=model.motion(d,d_mask)
            tcn=tcn.cpu().numpy();mt=mt.cpu().numpy();dt=dt.cpu().numpy()
        for b,i in enumerate(ids):
            r=rows[i];rk=r["raw_motion"]["sha256"];ak=r["audio"]["sha256"];full=sources[rk][2];av=audio[ak][0]
            me=selected[i]["music"];de=selected[i]["motion"]
            def pack(events,tokens,cap,fps,kind):
                x=np.zeros((cap,144),np.float32);valid=np.zeros(cap,bool);times=np.arange(cap,dtype=np.float32)/30+4.1
                n=len(events)
                if n:
                    t=[e["relative_seconds"] for e in events];x[:n,:128]=sample_tokens(tokens,t,fps);valid[:n]=True;times[:n]=t
                    for k,e in enumerate(events):
                        v=np.zeros(16,np.float32)
                        if kind=="music":v[:12]=av[e["frame"],18:30];v[12+("onset","accent","beat","silence").index(e["event_type"])]=1
                        else:
                            vector=full[e["frame"],:66].reshape(22,3)[list(PARTS[e["body_part"]])];speed=np.linalg.norm(vector,axis=-1)
                            v[:3]=vector.mean(0);v[3]=speed.mean();v[4]=speed.max();v[5+TYPES.index(e["event_type"])]=1
                            v[10+list(PARTS).index(e["body_part"])]=1;v[13]=e["confidence"]
                        x[k,128:]=(v-np.asarray(dn[kind]["mean"]))/np.asarray(dn[kind]["std"])
                return x,valid,times
            m30,mv,clock_m=pack(me,tcn[b],12,30,"music");m15,_,_=pack(me,tcn[b,::2],12,15,"music");mf,_,_=pack(me,mt[b],12,15,"music")
            df,dv,clock_d=pack(de,dt[b],20,15,"motion")
            for k,v in dict(music_tcn30=m30,music_tcn15=m15,music_transformer15=mf,motion=df,mv=mv,dv=dv,mt=clock_m,dt=clock_d).items():bank[k].append(v)
        if (start//32)%20==0:print("DENSE_ENCODED",start,len(rows),flush=True)
    bank={k:np.stack(v) for k,v in bank.items()};bank["music"]=bank["music_transformer15"]
    # Motion halos are not fake targets. The matching statistics only need the real core energy.
    stats_obs=dict(energy_music=arrays["energy_music"],energy_motion=np.stack([sources[r["raw_motion"]["sha256"]][2][r["start_frame"]:r["start_frame"]+150,66] for r in rows]))
    stats=nuisance_statistics(rows,bank,stats_obs,sidecars)
    conditional,evidence=fast_candidates(rows,bank,stats,config["strict"])
    np.savez_compressed(out/"nodes.npz",**bank);np.savez_compressed(out/"conditional.npz",**conditional)
    write(out/"candidate_evidence.json",evidence);write(out/"source_seals.json",list(seals.values()))
    write(out/"manifest.json",dict(schema="dense_source_clock_population_v1",rows=rows,full_source_records=len(sources),
        source_window_counts=source_sizes,stride_frames=config["stride_frames"],global_checkpoint=global_ref,
        source_population=ref(source/"data/manifest.json"),source_clock_manifest=ref(clock/"data/manifest.json"),
        current_generation_targets=False,new_jukebox_extracted=False,new_blind_source=False,strict_population=evidence["populations"],
        original_r1_thresholds_unchanged=True,windows_independent=False))
    return rows,bank,conditional,evidence

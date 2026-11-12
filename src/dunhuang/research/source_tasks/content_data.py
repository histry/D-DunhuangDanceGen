"""Fresh task-bound normalization of frozen real audio / full-source derivatives."""
import argparse,collections
from pathlib import Path
import numpy as np
from dunhuang.io import sha,checked,write
from dunhuang.data.sources import ref
from dunhuang.planning.time_map import motion_features74
from dunhuang.research.necessity.factors import donors
from dunhuang.alignment.sampling import group_schedule
from dunhuang.alignment.audio_clock import music_sample
from .common import read,commit,fresh,same_arrays,source_protocol

def swap_indices(rows):
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r["role"]=="val":groups[music_sample(r)].append(i)
    result=np.full(len(rows),-1,int)
    for ids in groups.values():
        for i in ids:
            candidates=[j for j in ids if rows[j]["music_uid"]!=rows[i]["music_uid"] and rows[j]["choreography_uid"]!=rows[i]["choreography_uid"]]
            if candidates:result[i]=min(candidates,key=lambda j:rows[j]["window_uid"])
    return result

def run(config,out,replay_from=None):
    out=fresh(out);root,_,_=source_protocol(config);source=Path(config["source_a2"]);published=read("reports/transfer_audit_summary.json")
    for name in ("manifest.json","train_statistics.json","source_seals.json"):
        if sha(source/name)!=published["evidence"]["a2/"+name]:raise ValueError("New task observation lineage differs")
    rows=read(source/"manifest.json")["rows"];stats=read(source/"train_statistics.json")
    if stats["test_tensor_reads"] or stats["statistics"]["music"]["scope"]!="Train-only full recordings/audio once per asset":raise ValueError("Fresh Train-only statistics absent")
    norm=dict(schema="content_task_normalizer_v1",population_binding=stats["population_binding"],norm=stats["statistics"],source_statistics=ref(source/"train_statistics.json"),test_tensor_reads=0,old_norm_reused=False)
    write(out/"normalizer.json",norm)
    arrays={name:np.empty((len(rows),120,c),dtype=dtype) for name,c,dtype in (("music",50,np.float32),("motion",74,np.float32),("mm",50,bool),("dm",74,bool))}
    audio={};last=None;continuous=None;observed=None;source_hashes={};audio_hashes={}
    for i,r in enumerate(rows):
        if r["role"] not in ("train","val"):raise ValueError("Test tensors forbidden")
        rk=r["source_native"]["sha256"];ak=r["audio"]["sha256"]
        if rk!=last:
            file=checked(r["source_native"]);native=np.load(file);continuous,observed=motion_features74(native);last=rk;source_hashes[rk]=sha(file)
        if ak not in audio:
            file=source/"audio"/(ak+".npz");other=Path(str(source)+"_replay")/"audio"/file.name
            if sha(file)!=sha(other):raise ValueError("Raw waveform observation replay hash differs")
            with np.load(file) as z:audio[ak]=(z["values"].copy(),z["mask"].copy())
            audio_hashes[ak]=sha(file)
        av,am=audio[ak];start=r["start_frame"]+15
        positions=start+30*r["audio_offset_seconds"]+np.arange(120);lo=np.floor(positions).astype(int);hi=np.ceil(positions).astype(int)
        if lo.min()<0 or hi.max()>=len(av) or start+120>len(continuous):raise ValueError("No audio/motion padding allowed")
        alpha=(positions-lo)[:,None]
        music=av[lo]*(1-alpha)+av[hi]*alpha;mm=am[lo]&am[hi];motion=continuous[start:start+120];dm=observed[start:start+120]
        for name,value,mask in (("music",music,mm),("motion",motion,dm)):
            mean=np.asarray(norm["norm"][name]["mean"]);std=np.asarray(norm["norm"][name]["std"])
            normalized=np.where(mask,(value-mean)/std,0).astype(np.float32)
            if not np.isfinite(normalized).all():raise ValueError("Nonfinite real content observations")
            arrays[name][i]=normalized;arrays["mm" if name=="music" else "dm"][i]=mask
        if i%2000==0:print("CONTENT_OBSERVATIONS",i,len(rows),flush=True)
    arrays["swap"]=swap_indices(rows);np.savez_compressed(out/"observations.npz",**arrays)
    candidates={};schedules=[];donor_ids=[];tasks=[];checks={}
    for task_id,task in enumerate(("time","association")):
        original=root/"a2"/(task+".npz")
        if sha(original)!=read("reports/source_necessity_summary.json")["evidence"]["a2/"+task+".npz"]:raise ValueError("Frozen A2 candidates differ")
        with np.load(original) as z:candidates[task]={k:z[k] for k in ("candidates","valid","positive","eligible")}
        pop=read(root/"a2/result.json")["tasks"][task]["populations"];checks[task]=all(p["population_sufficient"] for p in pop.values())
        ids=[i for i,r in enumerate(rows) if r["role"]=="train" and candidates[task]["eligible"][i]]
        local=group_schedule([rows[i] for i in ids],config["content"]["steps"]//2,config["content"]["batch"],config["seed"]+task_id)
        plan=np.asarray(ids)[local];wrong=donors(plan,rows,config["seed"]+task_id+41)
        schedules.append(plan);donor_ids.append(wrong);tasks.append(np.full(len(plan),task_id))
    plan=np.stack(schedules,1).reshape(-1,config["content"]["batch"]);wrong=np.stack(donor_ids,1).reshape(plan.shape)
    task_ids=np.stack(tasks,1).reshape(-1)
    np.savez_compressed(out/"schedule.npz",queries=plan,wrong_music=wrong,task=task_ids)
    write(out/"manifest.json",dict(rows=rows,candidate_files={t:ref(root/"a2"/(t+".npz")) for t in candidates}))
    gate=dict(qualified=all(checks.values()),checks=checks,scope=config["content"]["scope"],aliases="UNKNOWN",pretrained_exposure="UNKNOWN",operational_qualified=False)
    binding=sha(out/"normalizer.json")+":"+sha(out/"observations.npz")+":"+sha(out/"schedule.npz")
    write(out/"protocol.json",dict(code_commit=commit(),config=config,gate=gate,manifest=ref(out/"manifest.json"),normalizer=ref(out/"normalizer.json"),
        observations=ref(out/"observations.npz"),schedule=ref(out/"schedule.npz"),task_binding=binding,
        sources=source_hashes,audio=audio_hashes,music_operator="Real 30Hz full audio interpolated at exact source clock, core120",motion_operator="Full recording 74D derivatives before cropping core120",test_tensor_reads=0))
    report=dict(qualification=gate,windows=len(rows),train_windows=sum(r["role"]=="train" for r in rows),val_windows=sum(r["role"]=="val" for r in rows),
        native_sources=len(source_hashes),raw_audio_sources=len(audio_hashes),normalizer_bound=True,old_weights_reused=False,optimizer_calls=0,test_tensor_reads=0,operational_qualified=False)
    write(out/"result.json",report)
    if replay_from:
        old=Path(replay_from)
        if read(old/"result.json")!=report or read(old/"normalizer.json")!=norm:raise ValueError("Content task reconstruction differs")
        same_arrays(old/"observations.npz",out/"observations.npz");same_arrays(old/"schedule.npz",out/"schedule.npz")
        write(out/"replay.json",dict(status="FRESH_REAL_CONTENT_OBSERVATION_REPLAY",optimizer_calls=0,test_tensor_reads=0))
    print("CONTENT_TASK_GATE",gate,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

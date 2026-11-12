"""Real paired performances, original clocks, measured high-noise support."""
import argparse,itertools,sys,collections
import torch
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write,pinned
from dunhuang.data.sources import ref
from dunhuang.generation.targets import phase_target
from dunhuang.generation.source_pairs import validate_pair
from dunhuang.alignment.sampling import group_schedule,coverage
from .common import read,commit,fresh,source_protocol,same_arrays
from .math import angles,noise_support

def run(config,out,replay_from=None):
    out=fresh(out);root,p,m=source_protocol(config);cfg=config["g2"];assets=read(config["assets"])
    checked(assets["checkpoint"]);checked(assets["scaler"]);checked(m["initial_adapter"])
    sys.path.insert(0,str(pinned(assets["edge_root"],assets["edge_commit"])))
    from model.utils import make_beta_schedule
    alpha=torch.cumprod(1-torch.Tensor(make_beta_schedule("cosine",1000)),dim=0).numpy()
    masks=dict(np.load(checked(p["supervision"])));index={r["window_uid"]:r for r in m["rows"]}
    with np.load(checked(assets["scaler"])) as z:scale,bias=z["scale"],z["bias"]
    groups=collections.defaultdict(list)
    for r in m["rows"]:groups[(r["role"],r["choreography_uid"],r["music_uid"],r["audio"]["sha256"],r["start_frame"],r["audio_offset_seconds"])].append(r)
    cache={};verified={};pairs=[];rejections=collections.Counter()
    def check(r):
        key=(r["path"],r["sha256"])
        if key not in verified:verified[key]=checked(r)
        return verified[key]
    def data(uid):
        if uid not in cache:
            r=index[uid]
            for field in ("motion","observed","supervision","raw_motion","audio","jukebox","jukebox_receipt","alignment_attestation"):check(r[field])
            raw=np.load(check(r["motion"]));target,_=phase_target(raw,scale,bias)
            if target is None:raise ValueError("Full true target left frozen scaler support")
            proof=read(check(r["alignment_attestation"]))
            if proof["status"]!="verified" or proof["motion_sha256"]!=r["raw_motion"]["sha256"] or proof["audio_sha256"]!=r["audio"]["sha256"] or proof["audio_offset_seconds"]!=r["audio_offset_seconds"]:raise ValueError("Clock binding differs")
            if not np.load(check(r["supervision"])).all():raise ValueError("Unobserved complete target")
            cache[uid]=(raw,target)
        return cache[uid]
    for rs in groups.values():
        for ra,rb in itertools.combinations(sorted(rs,key=lambda r:r["window_uid"]),2):
            a,b=ra["window_uid"],rb["window_uid"]
            pair=dict(target_uid=a,source_uid=b,role=ra["role"],source_recording=rb["recording_uid"],target_recording=ra["recording_uid"],pose_exact_correspondence=False)
            validate_pair(pair,index)
            xa,ta=data(a);xb,tb=data(b)
            mask=masks[a+"__known"]&masks[b+"__known"]
            separation=float(angles(xa,xb,mask).mean())
            if separation<cfg["min_reference_separation_rad"]:rejections["source_not_separated"]+=1;continue
            if ra["jukebox"]["sha256"]!=rb["jukebox"]["sha256"]:rejections["actual_music_differs"]+=1;continue
            support,proof=noise_support(ta,tb,alpha,cfg)
            if len(support)<cfg["min_noise_support"]:rejections["insufficient_common_q_support"]+=1;continue
            pairs.append(dict(a=a,b=b,role=ra["role"],source_group=ra["source_group"],music_uid=ra["music_uid"],
                separation_rad=separation,ddim_support=support.tolist(),noise_proof=proof))
    train=[r for r in pairs if r["role"]=="train"];val=[r for r in pairs if r["role"]=="val"]
    tr_records={index[u]["recording_uid"] for r in train for u in (r["a"],r["b"])}
    val_records={index[u]["recording_uid"] for r in val for u in (r["a"],r["b"])}
    existing={(r["a"],r["b"]) for r in val}
    cases_ok=all((r["a"],r["b"]) in existing for r in m["validation_pairs"])
    checks=dict(train_pairs=len(train)>=cfg["min_train_pairs"],train_recordings=len(tr_records)>=cfg["min_train_recordings"],
        train_music=len({r["music_uid"] for r in train})>=cfg["min_train_music"],
        val_music=len({r["music_uid"] for r in val})>=cfg["min_val_music"],val_recordings=len(val_records)>=cfg["min_val_recordings"],
        frozen_four_cases=cases_ok,prior_source_replay=read(root/"preflight_replay/replay.json")["optimizer_calls"]==0)
    gate=dict(qualified=all(checks.values()),checks=checks,scope=cfg["scope"],aliases="UNKNOWN",pretrained_exposure="UNKNOWN",operational_qualified=False)
    if train:
        schedule=group_schedule(train,cfg["steps"],1,config["seed"]);rng=np.random.default_rng(config["seed"]+71)
        uniform=rng.random(cfg["steps"]);anchors=rng.integers(0,2,cfg["steps"])
        times=np.array([train[int(i)]["ddim_support"][int(u*len(train[int(i)]["ddim_support"]))] for i,u in zip(schedule[:,0],uniform)])
        cov=coverage(train,schedule)
    else:schedule=np.empty((0,1),int);times=np.empty(0,int);anchors=np.empty(0,int);cov={}
    np.savez_compressed(out/"schedule.npz",pairs=schedule,times=times,anchors=anchors,alpha=alpha)
    write(out/"pairs.json",dict(train=train,val=val,validation_pairs=m["validation_pairs"]))
    protocol=dict(schema="real_source_cf_qualification_v1",code_commit=commit(),config=config,gate=gate,
        previous_preflight=ref(root/"preflight/protocol.json"),pairs=ref(out/"pairs.json"),schedule=ref(out/"schedule.npz"),
        task_binding=sha(out/"pairs.json")+":"+sha(out/"schedule.npz"),source_targets="Each A/B target is its own real recording; never substitute one for the other",
        common_q="Equal mixture of q(x_t|X_A), q(x_t|X_B), one chosen draw shared by both branches; bounded conditional mean KL",
        pure_noise_used=False,optimizer_calls=0)
    write(out/"protocol.json",protocol)
    result=dict(qualification=gate,train_pairs=len(train),val_pairs=len(val),train_recordings=len(tr_records),val_recordings=len(val_records),
        train_music=len({r["music_uid"] for r in train}),val_music=len({r["music_uid"] for r in val}),pair_rejections=dict(rejections),
        schedule_coverage=cov,hash_checked_assets=len(verified),noise_support_sizes=dict(collections.Counter(str(len(r["ddim_support"])) for r in train)),
        max_accepted_mean_kl=max([r["noise_proof"]["max_selected_kl"] for r in pairs],default=0),test_tensor_reads=0,optimizer_calls=0,operational_qualified=False)
    write(out/"result.json",result)
    if replay_from:
        old=Path(replay_from)
        if read(old/"result.json")!=result or read(old/"pairs.json")!=read(out/"pairs.json"):raise ValueError("Real source qualification differs")
        same_arrays(old/"schedule.npz",out/"schedule.npz")
        write(out/"replay.json",dict(status="FRESH_REAL_PAIR_AND_Q_SUPPORT_REPLAY",optimizer_calls=0))
    print("REAL_SOURCE_PAIR_GATE",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

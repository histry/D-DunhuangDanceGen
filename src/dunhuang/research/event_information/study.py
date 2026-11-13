"""A2-P0: centre-vs-interval labels, pose information, no encoder/head training."""
import argparse,collections
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.data.conversion import decode
from dunhuang.data.change_event_phases import motion_profiles,detect
from dunhuang.research.source_capacity.common import read,commit,fresh,same_arrays,frozen
from dunhuang.research.reference_guidance.core import canonical,song_roles
from dunhuang.research.reference_guidance.events import aggregate,compare
from .operators import event_labels,phase_context,moments,finalize,normalize,neighbours,select_per_source,static_witness

def run(config,out,replay_from=None):
    out=fresh(out);root,pub,_,_=frozen(config);p=read(root/"content_data/protocol.json")
    manifest=read(checked(p["manifest"]));rows=manifest["rows"];checked(p["normalizer"]);arrays=dict(np.load(checked(p["observations"])))
    prior=read("reports/reference_guidance_summary.json");old=Path(config["reference_events"])/"events"
    if sha(old/"result.json")!=prior["evidence"]["events/result.json"] or sha(old/"scores.npz")!=prior["private_array_sha256"]["events/scores.npz"]:raise ValueError("Historical readout population differs")
    source=Path(config["source_a2"]);parent=read(source/"protocol.json");transfer=read("reports/transfer_audit_summary.json")
    for n in ("protocol.json","source_seals.json"):
        if sha(source/n)!=transfer["evidence"]["a2/"+n]:raise ValueError("Source audit seal differs")
    role_audit=song_roles(rows,read(checked(parent["registry"])),"AIST_strict_unseen_song")
    points=canonical(rows,"motion",96);old_scores=dict(np.load(old/"scores.npz"))
    if not np.array_equal(np.array([[i,t,f] for _,f,i,t in points]),old_scores["motion__indices"]):raise ValueError("Canonical physical-frame population changed")
    cfg=config["information"];phase_cfg=read("configs/data_event_v3.json")["phases"]
    by_source=collections.defaultdict(list)
    for n,(asset,frame,i,t) in enumerate(points):by_source[asset].append(n)
    pose=np.zeros((len(points),48),np.float32);pm=np.zeros_like(pose,bool);stage=np.zeros((len(points),4),np.float32);sm=np.zeros_like(stage,bool)
    truth=np.zeros((len(points),3),bool);stats={"pose":[],"stage":[]};events_refs={};raw_event_count=0;hold_span_frames=0;hold_centre_frames=0
    source_group=np.array([rows[i]["source_group"] for _,_,i,_ in points]);roles=np.array([rows[i]["role"] for _,_,i,_ in points])
    for asset,ids in sorted(by_source.items()):
        row=rows[points[ids[0]][2]];cached=np.load(checked(row["source_native"]));native,observed=decode(checked(row["raw_motion"]),"aistpp")
        observed=np.broadcast_to(observed,native.shape)
        if not np.array_equal(native,cached) or not observed[:,79:127].all():raise ValueError("Fresh raw source or observed upper8 differs")
        profile=motion_profiles(native,list(range(22)));events=detect(profile,phase_cfg)
        frames=np.array([points[n][1] for n in ids]);raw=native[:,79:127]
        # AIST++ source seal attests these fitted upper8 joints. Root/contact excluded.
        mask=observed[:,79:127].copy()
        sv,sk=phase_context(events,np.arange(len(native)),cfg["phase_time_clip_seconds"])
        pose[ids]=raw[frames];pm[ids]=mask[frames];stage[ids]=sv[frames];sm[ids]=sk[frames]
        truth[ids,0]=event_labels(events,frames,"upper_speed_peak",False,cfg["event_tolerance_frames"])
        truth[ids,1]=event_labels(events,frames,"upper_pose_hold",False,cfg["event_tolerance_frames"])
        truth[ids,2]=event_labels(events,frames,"upper_pose_hold",True)
        raw_event_count+=len(events);hold_span_frames+=int(event_labels(events,np.arange(len(native)),"upper_pose_hold",True).sum());hold_centre_frames+=int(event_labels(events,np.arange(len(native)),"upper_pose_hold",False).sum())
        if row["role"]=="train":
            stats["pose"].append(moments(raw,mask));stats["stage"].append(moments(sv,sk))
        events_refs[asset]=row["source_native"]
    if not np.array_equal(truth[:,0],old_scores["motion__labels"][:,1]) or not np.array_equal(truth[:,1],old_scores["motion__labels"][:,3]):raise ValueError("Frozen centre label replay changed")
    normalization={name:finalize(parts) for name,parts in stats.items()}
    pose,pm=normalize(pose,pm,normalization["pose"]);stage,sm=normalize(stage,sm,normalization["stage"])
    dynamic=np.stack([arrays["motion"][i,t] for _,_,i,t in points]);dm=np.stack([arrays["dm"][i,t] for _,_,i,t in points])
    permutation=np.arange(len(points));rng=np.random.default_rng(config["seed"])
    for asset,ids in sorted(by_source.items()):permutation[ids]=rng.permutation(ids)
    variants={"F0_F1":[(dynamic,dm)],"F2":[(dynamic,dm),(pose,pm)],"F3":[(dynamic,dm),(pose,pm),(stage,sm)],
        "F4_pose":[(dynamic,dm),(pose[permutation],pm[permutation]),(stage,sm)],"F4_time":[(dynamic,dm),(pose,pm),(stage[permutation],sm[permutation])]}
    bank=select_per_source(points,rows,"train",cfg["bank_per_train_source"]);train=select_per_source(points,rows,"train",cfg["queries_per_train_source"]);val=np.flatnonzero(roles=="val")
    saved=dict(labels=truth,bank=bank,train_queries=train,val_queries=val,indices=np.array([[i,t,f] for _,f,i,t in points]),pose=pose,pose_mask=pm,stage=stage,stage_mask=sm,permutation=permutation)
    metrics={};count=0
    labels={"speed_peak":0,"hold_centre":1,"hold_interval":2}
    for scope,qs in (("train_leave_song_out",train),("val_known_exposed_songs",val)):
        m={};records=[points[i] for i in qs]
        for variant,blocks in variants.items():
            ns=neighbours(blocks,qs,bank,source_group,cfg["k"]);saved[scope+"__"+variant+"__neighbours"]=ns
            for label,col in labels.items():
                if label=="hold_centre" and variant!="F0_F1":continue
                score=truth[ns,col].mean(1);saved[scope+"__"+variant+"__"+label]=score
                m[(variant,label)]=aggregate(truth[qs,col],score,rows,records,.5,"motion")
            print("INFORMATION_NEIGHBOURS",scope,variant,len(qs),flush=True)
        scope_results={}
        c=dict(min_val_groups=2,bootstrap_replicates=cfg["bootstrap_replicates"])
        for label in ("speed_peak","hold_interval"):
            scope_results[label]={variant:dict(vs_F1_dynamic=compare(m[(variant,label)],m[("F0_F1",label)],c),
                vs_F3=compare(m[(variant,label)],m[("F3",label)],c)) for variant in variants}
        scope_results["hold_centre_F0"]=compare(m[("F0_F1","hold_centre")],m[("F0_F1","hold_centre")],c)
        metrics[scope]=scope_results
    for key,stat in normalization.items():
        for n,v in stat.items():saved["normalizer__"+key+"__"+n]=v
    result=dict(metrics=metrics,role_audit=role_audit,population=dict(frames=len(points),train_bank=len(bank),train_queries=len(train),val_queries=len(val),source_recordings=len(by_source),train_music_groups=len(set(source_group[roles=="train"])),val_music_groups=len(set(source_group[roles=="val"]))),
        label_audit=dict(centre_positive=int(truth[:,1].sum()),interval_positive=int(truth[:,2].sum()),shared_positive=int((truth[:,1]&truth[:,2]).sum()),centre_positive_outside_interval=int((truth[:,1]&~truth[:,2]).sum()),interval_not_centre=int((~truth[:,1]&truth[:,2]).sum()),full_record_hold_span_frames=hold_span_frames,full_record_centre_positive_frames=hold_centre_frames,raw_detected_events=raw_event_count,
            different_targets_not_architecture_delta=True),
        static_witness=static_witness(),normalizer_fits=2,encoder_updates=0,head_fit_calls=0,optimizer_calls=0,nearest_neighbour_bank_is_train_only=True,
        phase_input_excludes_target_types=True,semantic_correspondence="NOT_EVALUABLE",human_reviewed=False,new_encoder_training_started=False,operational_qualified=False)
    np.savez_compressed(out/"arrays.npz",**saved);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,prepared=ref(root/"content_data/protocol.json"),frozen_labels=ref(old/"scores.npz"),registry=parent["registry"],raw_sources=events_refs,phase_config=ref("configs/data_event_v3.json"),
        existing74_normalizer=p["normalizer"],new_normalizers="Train full recordings once, observed upper8 pose and onset/turn timing only",
        phase_context="full-source offline onset/turn elapsed and future times; no hold or speed-peak label channels",
        input_block_metric="mean squared standardized distance over pairwise observed support, equal block weight",
        F0_vs_F1="different label definitions, no direct AP gain claim",split_scope="known publisher unseen-song; choreography exposed"))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result:raise ValueError("Information readouts differ")
        same_arrays(prev/"arrays.npz",out/"arrays.npz")
        write(out/"replay.json",dict(status="FRESH_RAW_INFORMATION_REPLAY",optimizer_calls=0,head_fit_calls=0,normalizers_recomputed=2,exact_arrays=True,nearest_neighbour_searches=10))
    print("EVENT_INFORMATION_AUDIT_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

"""A2: Train-only proxy decodability; not cross-modal semantic correspondence."""
import argparse,collections
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.data.change_event_phases import motion_profiles,detect
from dunhuang.generation.targets import digest
from dunhuang.research.source_tasks.content_model import ContentModel
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays,frozen,weight
from .core import canonical,ridge_fit,ridge_predict,group_scores,song_roles

KINDS={"music":["onset","beat"],"motion":["upper_motion_onset","upper_speed_peak","upper_direction_turn","upper_pose_hold"]}

def labels(points,rows,branch,source,phase_cfg,tolerance):
    events={};refs={};labels=[]
    for asset,frame,i,t in points:
        r=rows[i]
        if asset not in events:
            if branch=="music":
                file=source/"audio"/(asset+".json");other=Path(str(source)+"_replay")/"audio"/file.name
                if sha(file)!=sha(other):raise ValueError("Independent waveform event replay differs")
                j=read(file)
                if j["raw_audio"]!=r["audio"] or j["role"]!=r["role"]:raise ValueError("Raw audio event identity differs")
                checked(r["audio"])
                events[asset]=[dict(frame=e["frame"],type=e["event_type"]) for e in j["events"]]
                refs[asset]=ref(file)
            else:
                native=np.load(checked(r["source_native"]))
                # Differentiate/detect full recording, never a cropped window.
                events[asset]=detect(motion_profiles(native,list(range(22))),phase_cfg)
                refs[asset]=r["source_native"]
        labels.append([any(e["type"]==kind and abs(e["frame"]-frame)<=tolerance for e in events[asset]) for kind in KINDS[branch]])
    return np.asarray(labels,bool),refs

def aggregate(y,score,rows,points,threshold,branch):
    # Motion: each recording first, then equal music groups; audio deduplicated already.
    group=np.array([rows[i]["source_group"] for _,_,i,_ in points])
    record=np.array([rows[i]["recording_uid"] for _,_,i,_ in points])
    if branch=="music":return group_scores(y,score,group,threshold)
    raw=group_scores(y,score,record,threshold);result={}
    for g in sorted(set(group)):
        names=sorted(set(record[group==g]));values=[raw[n] for n in names if n in raw]
        if values:result[g]=np.mean(values,0).tolist()
    return result

def compare(a,b,cfg):
    common=sorted(set(a)&set(b))
    if len(common)<cfg["min_val_groups"]:return dict(status="NOT_EVALUABLE",groups=len(common))
    x=np.array([a[g] for g in common]);y=np.array([b[g] for g in common]);delta=x-y
    rng=np.random.default_rng(20261113);draws=delta[rng.integers(0,len(common),(cfg["bootstrap_replicates"],len(common)))].mean(1)
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    return dict(status="EXPOSED_VAL_PROXY_FRAME_DECODABILITY",groups=len(common),AP=float(x[:,0].mean()),F1=float(x[:,1].mean()),delta_AP=float(delta[:,0].mean()),delta_F1=float(delta[:,1].mean()),delta95=[lo.tolist(),hi.tolist()],both_lower_bounds_positive=bool((lo>0).all()))

def run(config,out,replay_from=None):
    out=fresh(out);root,pub,_,_=frozen(config);data=read(root/"content_data/protocol.json")
    rows=read(checked(data["manifest"]))["rows"];checked(data["normalizer"]);arrays=dict(np.load(checked(data["observations"])))
    source=Path(config["source_a2"]);transfer=read("reports/transfer_audit_summary.json")
    for name in ("protocol.json","source_seals.json","manifest.json","train_statistics.json"):
        if sha(source/name)!=transfer["evidence"]["a2/"+name]:raise ValueError("Data V3 source seal differs")
    source_protocol=read(source/"protocol.json");registry=read(checked(source_protocol["registry"]))
    role_audit=song_roles(rows,registry,"AIST_strict_unseen_song")
    if data["normalizer"]["sha256"]!=read(root/"content/protocol.json")["normalizer"]["sha256"]:raise ValueError("Diagnostic readout normalizer differs")
    phase_cfg=read("configs/data_event_v3.json")["phases"];cfg=config["events"];points={};truth={};inputs={};provenance={}
    # No model or score participates in choosing this unique physical-frame population.
    for branch in KINDS:
        pts=canonical(rows,branch,cfg[branch+"_cap_per_asset"])
        y,refs=labels(pts,rows,branch,source,phase_cfg,cfg["event_tolerance_frames"])
        points[branch]=pts;truth[branch]=y;provenance[branch]=refs
        inputs[branch]=np.stack([arrays[branch][i,t] for _,_,i,t in pts])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);forbid_optimizers()
    all_scores={};fits={};results={};fit_calls=0;weights={}
    population={}
    for branch,pts in points.items():
        roles=np.array([rows[i]["role"] for _,_,i,_ in pts]);groups=np.array([rows[i]["source_group"] for _,_,i,_ in pts])
        population[branch]=dict(frames=len(pts),train_frames=int((roles=="train").sum()),val_frames=int((roles=="val").sum()),train_groups=len(set(groups[roles=="train"])),val_groups=len(set(groups[roles=="val"])),assets=len({a for a,_,_,_ in pts}),positive_counts={kind:dict(train=int(truth[branch][roles=="train",k].sum()),val=int(truth[branch][roles=="val",k].sum())) for k,kind in enumerate(KINDS[branch])})
        all_scores[branch+"__labels"]=truth[branch]
        all_scores[branch+"__indices"]=np.array([[i,t,f] for _,f,i,t in pts])
    token_banks={}
    for arm in cfg["models"]:
        model=ContentModel("dual" if arm=="dual_shuffle" else arm).cuda();file=weight(root,pub,"content",arm)
        payload=torch.load(file,map_location="cuda",weights_only=True)
        if payload["task_binding"]!=data["task_binding"] or payload["normalizer_sha256"]!=data["normalizer"]["sha256"]:raise ValueError("Frozen representation binding differs")
        model.load_state_dict(payload["state"]);model.eval();model.requires_grad_(False);before=digest(model);weights[arm]=ref(file)
        token_banks[arm]={}
        with torch.no_grad():
            for branch,pts in points.items():
                needed=sorted({i for _,_,i,_ in pts});lookup={i:k for k,i in enumerate(needed)};tokens=[]
                for start in range(0,len(needed),32):
                    ids=needed[start:start+32]
                    v=getattr(model,branch)(torch.from_numpy(arrays[branch][ids]).cuda(),torch.from_numpy(arrays["mm" if branch=="music" else "dm"][ids]).cuda())
                    if not v[2].all():raise ValueError("Unobserved token in probe population")
                    tokens.append(v[1].cpu().numpy())
                bank=np.concatenate(tokens)
                token_banks[arm][branch]=np.stack([bank[lookup[i],t] for _,_,i,t in pts])
                print("EVENT_TOKENS",arm,branch,len(pts),flush=True)
        if digest(model)!=before:raise ValueError("Encoder parameters changed")
        del model;torch.cuda.empty_cache()
    for branch,pts in points.items():
        roles=np.array([rows[i]["role"] for _,_,i,_ in pts]);groups=np.array([rows[i]["source_group"] for _,_,i,_ in pts])
        train=roles=="train";val=roles=="val";x=inputs[branch];y=truth[branch]
        time=np.array([f/30 for _,f,_,_ in pts])[:,None]
        simple=np.c_[time,x[:,[32,33,34]]] if branch=="music" else np.c_[time,x[:,[66,67,70,72,73]]]
        # Independent within-asset permutation for each role: preserves source/class marginals.
        perm=np.arange(len(pts));rng=np.random.default_rng(config["seed"])
        buckets=collections.defaultdict(list)
        for k,(asset,_,i,_) in enumerate(pts):buckets[(roles[k],asset)].append(k)
        for ids in buckets.values():perm[ids]=rng.permutation(ids)
        features={"simple_time_signal":simple,"raw_observations":x}
        for arm in cfg["models"]:features[arm]=token_banks[arm][branch];features[arm+"_permuted"]=token_banks[arm][branch][perm]
        results[branch]={}
        for k,kind in enumerate(KINDS[branch]):
            target=y[:,k];tasks={};group_results={}
            if target[train].sum()<cfg["min_train_positive"] or (~target[train]).sum()<cfg["min_train_positive"] or target[val].sum()<cfg["min_val_positive"]:
                results[branch][kind]=dict(status="NOT_EVALUABLE",reason="insufficient fixed positive/negative frame support");continue
            for name,feature in features.items():
                key=branch+"__"+kind+"__"+name
                if replay_from:
                    oldfits=np.load(Path(replay_from)/"fits.npz")
                    fit={n:oldfits[key+"__"+n] for n in ("mean","std","beta")}
                else:fit=ridge_fit(feature[train],target[train],groups[train],cfg["ridge"]);fit_calls+=1
                for n,value in fit.items():fits[key+"__"+n]=value
                score=ridge_predict(fit,feature);all_scores[key]=score
                vp=[p for p,take in zip(pts,val) if take]
                group_results[name]=aggregate(target[val],score[val],rows,vp,cfg["threshold"],branch)
            for name,g in group_results.items():
                tasks[name]=dict(vs_simple=compare(g,group_results["simple_time_signal"],cfg),vs_raw=compare(g,group_results["raw_observations"],cfg))
                if name in cfg["models"]:
                    tasks[name]["vs_permuted"]=compare(g,group_results[name+"_permuted"],cfg)
                    # Same trained head, permute ONLY Val content (zero new fitting).
                    key=branch+"__"+kind+"__"+name;fit={n:fits[key+"__"+n] for n in ("mean","std","beta")}
                    swap=ridge_predict(fit,features[name][perm]);all_scores[key+"__val_swap"]=swap
                    swapped=aggregate(target[val],swap[val],rows,vp,cfg["threshold"],branch)
                    tasks[name]["vs_val_feature_swap"]=compare(g,swapped,cfg)
            results[branch][kind]=tasks
    np.savez_compressed(out/"scores.npz",**all_scores);np.savez_compressed(out/"fits.npz",**fits)
    result=dict(population=population,role_audit=role_audit,probes=results,linear_fit_calls=read(Path(replay_from)/"result.json")["linear_fit_calls"] if replay_from else fit_calls,
        replay_fit_calls=0,optimizer_calls=0,encoder_updates=0,human_reviewed=False,semantic_event_truth=False,
        cross_modal_event_correspondence="NOT_EVALUABLE_WITHOUT_INDEPENDENT_EVENT_TRUTH",full_timeline_localization="NOT_EVALUABLE_SAMPLED_FRAME_PROTOCOL",
        development_exposed=True,operational_qualified=False)
    write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,prepared=ref(root/"content_data/protocol.json"),weights=weights,provenance=provenance,
        role_registry=source_protocol["registry"],role_audit=role_audit,phases=phase_cfg,phase_config=ref("configs/data_event_v3.json"),population_selection="unique physical frames, uniform per-asset caps, nearest-context centre, no labels/scores",
        probe_fit="Train-only source-group and class-balanced ridge; fixed lambda and threshold",
        proxy_limit="music50 contains audio-derived controls; motion74 contains kinematics; decoding these proxy labels is not semantic understanding"))
    if replay_from:
        old=Path(replay_from)
        if read(old/"result.json")!=result:raise ValueError("Event readouts differ")
        same_arrays(old/"scores.npz",out/"scores.npz");same_arrays(old/"fits.npz",out/"fits.npz")
        write(out/"replay.json",dict(status="FRESH_FROZEN_EVENT_READOUT_REPLAY",optimizer_calls=0,linear_fit_calls=0,exact_arrays=True,models_verified=4))
    print("EVENT_INFORMATION_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

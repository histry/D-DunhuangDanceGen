"""Full-source qualification before model outputs; all exclusions remain visible."""
import argparse,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import soundfile as sf
from dunhuang.io import checked,write,sha
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter
from dunhuang.research.source_capacity.common import read,commit,fresh,same_arrays,forbid_optimizers
from dunhuang.research.masked_denoiser.common import assets_and_population
from dunhuang.research.global_clock_r17.contract import pair_candidates
from dunhuang.research.global_clock_r17.data import audit_pair
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from .contract import support_proxy

def same_audio_pcm(a,b):
    x,sr=sf.read(a,dtype="float32",always_2d=True);y,other=sf.read(b,dtype="float32",always_2d=True)
    return sr==other and x.shape==y.shape and np.array_equal(x,y)

def rank(cfg,text):return hashlib.sha256((str(cfg["seed"])+text).encode()).hexdigest()

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers()
    assert sha(Path(cfg["r20_root"])/"interface/protocol.json")==read("reports/r20_release/r20_receipt.json")["evidence"]["interface/protocol.json"]
    assets,rows,_,_,_=assets_and_population(read("src/dunhuang/research/masked_denoiser/config.json"))
    native_environment(assets)
    baseline_report=read("reports/r20_release/r20_summary.json")["native"]["structure"][0]["native_report"]
    native_limits=read(checked(baseline_report))["absolute"]["limits"]
    assert native_limits["joint_jerk_mps3_max"]==cfg["fixed_jerk_limit"]
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS,OFFSETS
    scaler=np.load(checked(assets["scaler"]));full_cache={};eligible=defaultdict(list);ledger=[];counts=Counter()
    for left,right in pair_candidates(rows):
        key=left["source_native"]["sha256"]
        if key not in full_cache:
            x=np.load(checked(left["source_native"]));p=fk_24_np(x)
            full_cache[key]=(x,p,support_proxy(p,cfg["proxy"]))
        full,p,proxy=full_cache[key];a=left["start_frame"];x=full[a:a+225]
        origin=x[0,4:7].copy();origin[1]=0.
        encoded=MotionAdapter.to_edge(x,origin)*scaler["scale"]+scaler["bias"]
        jerk=np.linalg.norm(np.diff(p[a:a+225],n=3,axis=0)*30.**3,axis=-1)
        leftmax=max(v[0] for v in cfg["contexts"]);rightmax=max(v[1] for v in cfg["contexts"])
        fixed=(np.arange(222)+3<leftmax)|(np.arange(222)>=225-rightmax)
        bound=float(jerk[fixed].max())
        support=bool((abs(encoded[:,4:])<=1+cfg["scaler_epsilon"]*np.finfo(np.float32).eps).all())
        status="ELIGIBLE" if support and bound<=cfg["fixed_jerk_limit"] else "OUT_OF_SCALER" if not support else "EXCLUDED_IMMUTABLE_JERK"
        item=dict(source_group=left["source_group"],recording_uid=left["recording_uid"],start_frame=a,
            raw_motion_sha256=left["raw_motion"]["sha256"],source_native_sha256=key,status=status,
            immutable_jerk_lower_bound=bound,scaler_support=support)
        ledger.append(item);counts[status]+=1
        if status=="ELIGIBLE":eligible[left["source_group"]].append((left,right))
    groups=sorted([g for g,p in eligible.items() if len({l["recording_uid"] for l,r in p})>=2],key=lambda g:rank(cfg,g))
    # Preserve the two R20 exposed DEV groups; never adapt on them in R21.
    mandatory=["aist_publisher_track:mLO2","aist_publisher_track:mHO1"]
    assert all(g in eligible for g in mandatory)
    dev=mandatory+[g for g in groups if g not in mandatory and g not in ("aist_publisher_track:mWA1","aist_publisher_track:mJS1")][:cfg["dev_groups"]-len(mandatory)]
    warm=["aist_publisher_track:mWA1","aist_publisher_track:mJS1"]
    assert all(g in groups for g in warm)
    train=warm+[g for g in groups if g not in dev and g not in warm][:cfg["train_groups"]-len(warm)]
    assert len(dev)==cfg["dev_groups"] and len(train)==cfg["train_groups"] and not set(train)&set(dev)
    selected=[]
    for role,gs,n in (("train",train,cfg["train_recordings_per_group"]),("dev_exposed",dev,cfg["dev_recordings_per_group"])):
        for group in gs:
            recordings=defaultdict(list)
            for l,r in eligible[group]:recordings[l["recording_uid"]].append((l,r))
            chosen=sorted(recordings,key=lambda x:rank(cfg,x))[:n]
            for recording in chosen:
                pair=min(recordings[recording],key=lambda lr:rank(cfg,lr[0]["raw_motion"]["sha256"]+str(lr[0]["start_frame"])))
                selected.append((role,pair))
    records=[];parity={}
    for case,(role,(left,right)) in enumerate(selected):
        folder=out/str(case);folder.mkdir()
        row=audit_pair(left,right,assets,folder/"data.npz")
        full,p,proxy=full_cache[left["source_native"]["sha256"]];a=left["start_frame"]
        z=dict(np.load(checked(row["arrays"])))
        z.update({name:value if name=="proxy_floor" else value[a:a+225] for name,value in proxy.items()})
        np.savez_compressed(folder/"data.npz",**z);row["arrays"]=ref(folder/"data.npz")
        gt,physical=audit_native(z["native"]);assert physical["absolute"]["limits"]==native_limits
        verified_jerk=np.linalg.norm(np.diff(gt,n=3,axis=0)*30.**3,axis=-1)
        fixed=(np.arange(222)+3<leftmax)|(np.arange(222)>=225-rightmax)
        assert float(verified_jerk[fixed].max())<=cfg["fixed_jerk_limit"]
        write(folder/"source_physical.json",physical);parity[str(case)]=gt
        row.update(case=case,role=role,source_physical=ref(folder/"source_physical.json"),proxy_scope="SOURCE_GEOMETRY_PROXY_NOT_CONTACT_TRUTH",
            support_frames=int(z["proxy_static"].sum()),swing_frames=int(z["proxy_swing"].sum()),
            full_source_floor=float(z["proxy_floor"]),full_recording_frames=len(full),
            source_activity_mps=float(np.linalg.norm(np.diff(gt,axis=0)*30,axis=-1).mean()),
            historical_exposure="DEVELOPMENT_EXPOSED_OR_UNKNOWN_PRETRAINING; not blind")
        records.append(row)
        if replay:
            old=read(Path(replay)/"result.json")["records"][case]
            same_arrays(checked(row["arrays"]),checked(old["arrays"]))
            assert same_audio_pcm(checked(row["audio225"]),checked(old["audio225"]))
            assert physical==read(checked(old["source_physical"]))
    np.savez_compressed(out/"parity.npz",parents=PARENTS,offsets=OFFSETS,**parity)
    for field in ("audio","raw_motion","source_native"):
        train_ids={r["left"][field]["sha256"] for r in records if r["role"]=="train"}
        dev_ids={r["left"][field]["sha256"] for r in records if r["role"]=="dev_exposed"}
        assert not train_ids&dev_ids,("Cross-role identity",field)
    assert not {r["left"]["recording_uid"] for r in records if r["role"]=="train"}&{r["left"]["recording_uid"] for r in records if r["role"]=="dev_exposed"}
    train_cases=[r["case"] for r in records if r["role"]=="train"]
    dev_cases=[r["case"] for r in records if r["role"]=="dev_exposed"]
    # One recording from each of the first four fixed Train music groups.
    probe=[];probe_groups=[]
    for group in train:
        candidates=[r for r in records if r["role"]=="train" and r["left"]["source_group"]==group and r["support_frames"]>0]
        if candidates:probe.append(candidates[0]["case"]);probe_groups.append(group)
        if len(probe)==4:break
    for group in train:
        if len(probe)==4:break
        if group not in probe_groups:
            probe.append(next(r["case"] for r in records if r["role"]=="train" and r["left"]["source_group"]==group));probe_groups.append(group)
    support_cases=sum(next(r for r in records if r["case"]==c)["support_frames"]>0 for c in probe)
    result=dict(records=records,train_cases=train_cases,dev_cases=dev_cases,train_probe=probe,follow_cases=probe,
        train_groups=train,dev_groups=dev,candidate_counts=dict(counts),candidate_ledger=ledger,
        support_admitted=support_cases>=2,proxy_support_cases=support_cases,probe_selection="Pre-model source-proxy-presence stratification; one recording per first four eligible Train groups; no generated metrics",paired_data_ready=True,
        source_native_passes=sum(read(checked(r["source_physical"]))["absolute"]["ok"] for r in records),
        audio_replay_contract="Each container hash authenticated independently; rate/channels/PCM exact; WAV PEAK write timestamps may differ",optimizer_steps=0,whole_song_generations=0,operational_qualified=False)
    if replay:
        old=read(Path(replay)/"result.json")
        for n in ("candidate_ledger","train_cases","dev_cases","train_probe","support_admitted","train_groups","dev_groups"):assert result[n]==old[n],n
        same_arrays(out/"parity.npz",Path(replay)/"parity.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,assets=assets,parity=ref(out/"parity.npz"),
        source_population="R12 qualified fit rows; all global Data V3 Train; R21 adaptation groups disjoint",
        proxy_method="Full recording FK/difference/floor/run anchors before window slicing",
        fixed_exclusion_is_not_sufficient_feasibility=True,pretrained_exposure="UNKNOWN"))
    print("R21_DATA",dict(counts),"train",len(train_cases),"dev",len(dev_cases),"support_admitted",result["support_admitted"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.out,a.replay_from)

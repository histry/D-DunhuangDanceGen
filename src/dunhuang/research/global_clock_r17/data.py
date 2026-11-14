"""Source-authenticated, group-separated continuous225 targets before training."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter
from dunhuang.data.conversion import decode
from dunhuang.research.source_capacity.common import read,commit,fresh,same_arrays
from dunhuang.research.masked_denoiser.common import assets_and_population
from dunhuang.research.continuous_r16.contract import continuous_pair
from dunhuang.research.state_bridge.common import native_environment
from .contract import views,feature_distance,metadata

def audit_pair(left,right,assets,path):
    continuous_pair(left,right);full=np.load(checked(left["source_native"]));raw,_=decode(checked(left["raw_motion"]),"aistpp")
    assert np.array_equal(raw,full)
    a=left["start_frame"];x=full[a:a+225]
    if len(x)!=225 or a<8 or a+158>len(full):raise ValueError("Missing real continuous source")
    clock=read(checked(left["clock"]));cal=clock["calibration"]
    assert clock["status"]=="VERIFIED_PAIR" and cal["verified"]
    assert clock["raw_motion"]["sha256"]==left["raw_motion"]["sha256"] and clock["raw_audio"]["sha256"]==left["audio"]["sha256"]
    features=[];audio=[];producer=[]
    for row in (left,right):
        p=Path(checked(row["jukebox"])).parent/"producer.json";pr=read(p);producer.append(ref(p))
        expected=round((cal["audio_offset_seconds"]+row["start_frame"]/30)*pr["rate"])
        assert pr["sample_start"]==expected and pr["sample_end"]==expected+5*pr["rate"]
        assert pr["audio"]==row["audio"]["sha256"] and pr["features_sha256"]==row["jukebox"]["sha256"] and pr["layer"]==66 and pr["fps"]==30
        f=np.load(checked(row["jukebox"]));assert f.shape==(150,4800) and np.isfinite(f).all();features.append(f)
        wav,sr=sf.read(checked(pr["crop"]),dtype="float32",always_2d=True);assert sr==pr["rate"];audio.append(wav)
    assert read(checked(producer[0]))["rate"]==read(checked(producer[1]))["rate"]
    half=round(2.5*sr);assert np.array_equal(audio[0][half:],audio[1][:half])
    m=np.concatenate((features[0],features[1][75:]))
    scaler=np.load(checked(assets["scaler"]));origin=x[0,4:7].copy();origin[1]=0
    encoded=MotionAdapter.to_edge(x,origin)*scaler["scale"]+scaler["bias"]
    if (abs(encoded[:,4:])>1+32*np.finfo(np.float32).eps).any():raise ValueError("Selected225 target outside frozen support; do not replace population")
    np.savez_compressed(path,native=x,encoded=encoded,music=m,original_music=np.stack(features),meta=metadata(),
        halo150=full[a-8:a+158],origin=origin)
    wavfile=path.with_suffix(".wav");sf.write(wavfile,np.concatenate((audio[0],audio[1][half:])),sr,subtype="FLOAT")
    return dict(left=left,right=right,arrays=ref(path),audio225=ref(wavfile),producer=producer,
        fresh_raw_decode_exact=True,audio_overlap_exact=True,feature_overlap=feature_distance(features[0][75:],features[1][:75]),
        music_policy="Earlier authenticated crop owns repeated global time, not full-context extraction",global_scaler_support=True)

def run(cfg,out,replay=None):
    out=fresh(out);r16=Path(cfg["r16_root"]);pub=read("reports/r16_summary.json")
    assert sha("reports/r16_summary.json")==read("reports/r16_receipt.json")["summary_sha256"]
    for name in ("preflight/protocol.json","preflight/result.json"):assert sha(r16/name)==pub["evidence"][name]
    prior=read(r16/"preflight/protocol.json");eval_pairs=[(b["left"],b["right"]) for b in prior["pairs"]]
    assets,rows,_,_,_=assets_and_population(read(cfg["r13_config"]))
    by={(r["recording_uid"],r["start_frame"]):r for r in rows}
    oldgroups={l["source_group"] for l,r in eval_pairs};pairs=[]
    for l in rows:
        r=by.get((l["recording_uid"],l["start_frame"]+75))
        if r and l["start_frame"]>=8 and l["source_group"] not in oldgroups:pairs.append((l,r))
    groups=sorted({l["source_group"] for l,r in pairs},key=lambda g:hashlib.sha256((str(cfg["seed"])+g).encode()).hexdigest())[:cfg["fit_groups"]]
    selected=[min((p for p in pairs if p[0]["source_group"]==g),key=lambda p:hashlib.sha256((str(cfg["seed"])+p[0]["raw_motion"]["sha256"]+str(p[0]["start_frame"])).encode()).hexdigest()) for g in groups]
    assert len(groups)==8 and len(eval_pairs)==4 and not set(groups)&oldgroups
    native_environment(assets)
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS,OFFSETS
    records=[];parity={}
    for role,pop in (("fit",selected),("eval",eval_pairs)):
        for i,(l,r) in enumerate(pop):
            name=role+"_"+str(i);record=audit_pair(l,r,assets,out/(name+".npz"));record.update(role=role,case=i)
            records.append(record);parity[name]=fk_24_np(np.load(checked(record["arrays"]))["native"])
    np.savez_compressed(out/"parity.npz",parents=PARENTS,offsets=OFFSETS,**parity)
    result=dict(records=records,fit_groups=groups,eval_groups=sorted(oldgroups),continuous_data_ready=True,
        task_scopes=dict(free="No future/root full target observations",inbetween="Real observed boundary states",
            progressive="Only a hash-bound previously generated prefix; no future full target motion"),
        c2_minimal_training_registered=True,old_r16_gate_promoted=False,operational_qualified=False,optimizer_steps=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,assets=assets,parity=ref(out/"parity.npz"),
        source_population="R13 fit groups, adapter-fit8 disjoint from four exposed R16 evaluation groups"))
    if replay:
        old=read(Path(replay)/"result.json")
        for a,b in zip(records,old["records"]):
            same_arrays(checked(a["arrays"]),checked(b["arrays"]));assert a["feature_overlap"]==b["feature_overlap"]
        same_arrays(out/"parity.npz",Path(replay)/"parity.npz")
    print("R17_DATA",len(records),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

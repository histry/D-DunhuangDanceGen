"""Freeze Train-only objective scales, disjoint probe identities and real music."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.data.split_registry import stable
from dunhuang.research.source_identifiability.g2 import make_condition
from dunhuang.research.reference_guidance.core import phase_reference
from .common import read,commit,fresh,prior,frozen,selected_pairs
from .losses import scales

def run(config,out,replay_from=None):
    out=fresh(out);old,public=prior(config);root,pub,p,manifest=frozen(config);index={r["window_uid"]:r for r in manifest["rows"]}
    calibration,new=selected_pairs(config,manifest);available=[];conditions=[]
    phase_cfg=read("configs/data_event_v3.json")["phases"]
    for pair in calibration:
        for key in ("a","b"):
            uid=pair[key];condition,keys,plan,contract=make_condition(index[uid],manifest["audio_contract"])
            available.append(phase_reference(condition.value[0].numpy(),condition.known[0].numpy(),keys[0].numpy(),phase_cfg))
            conditions.append(dict(window_uid=uid,raw_motion=index[uid]["raw_motion"],observed=index[uid]["observed"],plan=plan,contract=contract))
    norm=scales(available,config["probe"]);write(out/"normalizer.json",norm);write(out/"calibration_sources.json",conditions)
    rows=read(checked(read(root/"content_data/protocol.json")["manifest"]))["rows"];music={}
    old_audio={v["payload"]["audio_sha256"] for v in read(Path(config["data_v3"])/"main/protocol.json")["timelines"]}
    for r in rows:
        if r["role"]=="train":music.setdefault(r["audio"]["sha256"],dict(audio=r["audio"],work_uid=r["source_group"],role=r["role"]))
    selected=[];genres=set();pcm=set()
    for key,value in sorted(music.items(),key=lambda kv:stable([config["selection_seed"],kv[0]])):
        if key in old_audio:continue
        genre=value["work_uid"].split(":")[-1][1:3]
        if genre in genres:continue
        path=checked(value["audio"]);info=sf.info(path)
        if info.duration<config["music"]["frames"]/30:continue
        wave,sr=sf.read(path,dtype="float32",always_2d=True)
        digest=hashlib.sha256(np.ascontiguousarray(wave).tobytes()+str((sr,wave.shape)).encode()).hexdigest()
        if digest in pcm:continue
        selected.append(dict(value,pcm_sha256=digest,sample_rate=sr,samples=len(wave),channels=wave.shape[1],duration_seconds=len(wave)/sr,genre_proxy=genre));genres.add(genre);pcm.add(digest)
        if len(selected)==config["music"]["count"]:break
    if len(selected)!=config["music"]["count"]:raise ValueError("Distinct real Train music population insufficient")
    write(out/"music.json",selected);write(out/"pairs.json",dict(calibration=calibration,disjoint_probe=new))
    old_shas={index[p[k]]["raw_motion"]["sha256"] for p in calibration for k in ("a","b")}
    new_shas={index[p[k]]["raw_motion"]["sha256"] for p in new for k in ("a","b")}
    if old_shas&new_shas or {p["source_group"] for p in calibration}&{p["source_group"] for p in new}:raise ValueError("Probe population overlaps calibration")
    result=dict(calibration_pairs=len(calibration),disjoint_probe_pairs=len(new),raw_source_overlap=0,music_group_overlap=0,
        disjoint_probe_is_not_unseen_training_source=True,source_training_exposure="EXPOSED_TRAIN_CHECKPOINT_POPULATION",fresh_blind=False,
        real_music_assets=len(selected),distinct_pcm=len(pcm),distinct_publisher_work_ids=len({r["work_uid"] for r in selected}),
        music_independence_scope="known publisher track identity, composition aliases outside publisher UNKNOWN",
        normalizer_fit_calls=1,model_fit_calls=0,test_tensor_reads=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=config,normalizer=ref(out/"normalizer.json"),pairs=ref(out/"pairs.json"),music=ref(out/"music.json"),calibration_sources=ref(out/"calibration_sources.json"),source_manifest=p["manifest"],old_source_result=ref(old/"protected/result.json"),
        temporal_reference="provided visible adjacent poses only; no differentiation across visibility gaps",candidate_selection="seeded hash and distinct publisher genres, no model scores"))
    if replay_from:
        prev=Path(replay_from)
        for name in ("normalizer.json","pairs.json","music.json","result.json","calibration_sources.json"):
            if read(prev/name)!=read(out/name):raise ValueError("Frozen research population/scales differ")
        write(out/"replay.json",dict(status="FRESH_RESEARCH_POPULATION_REPLAY",optimizer_calls=0,exact_metadata=True,test_tensor_reads=0))
    print("R6_PREPARATION_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

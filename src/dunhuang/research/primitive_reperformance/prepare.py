"""Freeze source-disjoint complete events before any new model generation."""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.planning.feasible_time_map import solve
from dunhuang.planning.event_plan import validate
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.source_expression.review import audio_clip

def stable(seed,value):return hashlib.sha256((str(seed)+value).encode()).hexdigest()

def event(old,new,T,beats):
    c=new["safe_anchor"]["frame"]
    solution=solve(dict(source_interval=new["source_interval"],center=c,normalization_support=new["normalization_support"]),dict(start=0,end=T),beats)
    if solution["status"]!="PASS":return None
    e=copy.deepcopy(old);e.update(center=c,music_interval=[0,T],time_warp=solution["time_warp"],
        primitive=dict(source_frame=c,music_frame=solution["tick"],joints=list(range(12,20)),radius=12),
        observed_joints=new["observed_joints"],root_translation_observed=False,contact_observed=False,
        full_phases=new["kinematic_phases"],key_motion=new["key_motion"])
    return e

def make_plan(e,T,audio):
    plan=dict(schema="event_plan_v1",fps=30,frames=T,audio_sha256=audio["sha256"],time_scale_limits=[.75,1.33],events=[e])
    validate(plan,audio["sha256"],T);return plan

def run(cfg,out):
    out=fresh(out);prior=Path(cfg["prior_song"]);pub=read("reports/exposed_full_song_summary.json")
    for phase in ("music","features"):
        if sha(prior/phase/"protocol.json")!=pub["evidence"][phase+"/protocol.json"]:raise ValueError("Prior full-song seal differs")
    parent=read(Path(cfg["data_v3"])/"main/protocol.json");catalog=read(checked(parent["catalog"]))
    e1file=Path(cfg["data_v3"])/"main/event_catalog_v2_shadow.json"
    public=read("reports/data_event_v3_summary.json")
    if sha(e1file)!=public["evidence"]["main/event_catalog_v2_shadow.json"]:raise ValueError("E1 phase sidecar seal differs")
    e1=read(e1file)["entries"];old={e["event_uid"]:e for e in catalog["entries"]}
    groups=sorted({e["source_uid"] for e in e1});assert len(groups)==2*cfg["pairs"]
    eligible=[e for e in e1 if e["safe_anchor"] and e["normalization_support"] and cfg["event_duration"][0]<=e["duration_frames"]<=cfg["event_duration"][1]
        and e["observed_joints"]==list(range(22)) and len({p["frame"] for p in e["kinematic_phases"]})>=cfg["minimum_phases"]]
    features=read(prior/"features/protocol.json");music=[]
    for k in cfg["music_cases"]:
        s=features["songs"][k];x=np.load(checked(s["features"]));file=out/("music_"+str(k)+".npy");np.save(file,x[:1])
        wave,sr=sf.read(checked(s["audio"]),dtype="float32",always_2d=True);file_audio=out/("music_"+str(k)+".wav");audio_clip(file_audio,wave[:5*sr].mean(1)[:,None],sr)
        music.append(dict(case=k,name=Path(s["audio"]["path"]).name,audio=ref(file_audio),features=ref(file),full_original_audio=s["audio"],feature_producer=ref(prior/"features/protocol.json")))
    beats=read(prior/"music/0_acoustic_v1_timeline.json")["beats"];pairs=[]
    for i in range(cfg["pairs"]):
        left=sorted([e for e in eligible if e["source_uid"]==groups[i]],key=lambda e:stable(cfg["seed"],e["event_uid"]))
        right=sorted([e for e in eligible if e["source_uid"]==groups[i+cfg["pairs"]]],key=lambda e:stable(cfg["seed"],e["event_uid"]))
        selected=None
        for a in left:
            for b in right:
                lower=max(75,(75*max(a["duration_frames"],b["duration_frames"])+99)//100)
                upper=min(150,133*min(a["duration_frames"],b["duration_frames"])//100)
                for T in sorted(range(lower,upper+1),key=lambda t:(abs(t-cfg["target_frames"]),t)):
                    A=event(old[a["event_uid"]],a,T,beats);B=event(old[b["event_uid"]],b,T,beats)
                    if A and B:selected=(A,B,T);break
                if selected:break
            if selected:break
        if selected is None:raise ValueError("No complete legal source pair; no outcome-dependent replacement")
        A,B,T=selected;plans={}
        for label,e in (("A",A),("B",B)):
            file=out/("pair_"+str(i)+"_"+label+".json");write(file,make_plan(e,T,music[0]["audio"]));plans[label]=ref(file)
        pairs.append(dict(pair=i,cohort="calibration" if i<cfg["pairs"]//2 else "source_disjoint_probe",frames=T,plans=plans,
            source_uids=[A["source_uid"],B["source_uid"]],source_sha256=[A["source_sha256"],B["source_sha256"]]))
    write(out/"pairs.json",pairs);write(out/"music.json",music)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,pairs=ref(out/"pairs.json"),music=ref(out/"music.json"),
        catalog=parent["catalog"],phase_catalog=ref(e1file),assets=ref(cfg["assets"]),
        selection="first hash-ordered legal full Core pair in each fixed source pair; no Native/outcome filtering",
        reference_clock="music0 beat clock held fixed across music swap to isolate generator content response",
        independent_unseen_semantic_evaluation=False,neural_training_updates=0))
    print("R8_FULL_EVENT_PREPARED",len(pairs),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(read(a.config),a.out)

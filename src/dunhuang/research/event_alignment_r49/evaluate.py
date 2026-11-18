"""Historical Val evaluation by recording and song, with explicit blind-test status."""
from collections import defaultdict
from pathlib import Path
import numpy as np

from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read, sha, write
from dunhuang.research.music_phase_r47.phase_audit import stage_score
from dunhuang.research.sdmr_event_rank.evaluate import group_ci
from .common import Frozen, OFFSETS, RADIUS, SEED, audio_signals, cosine, interpolate
from .rank import PREVIEW_ROOT

AF = Path("outputs/transfer_audit_01/a2/audio")


def hit(scores, positive):
    scores = np.asarray(scores, dtype=float)
    tied = np.isclose(scores, scores.max(), rtol=0, atol=1e-7)
    return float(tied[positive]/tied.sum())


def summaries(tasks, actual="Content"):
    by_method = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in tasks:
        for method, score in row["scores"].items():
            by_method[method][row["audio"]][row["recording"]].append(hit(score,row["positive"]))
    means = {m:{g:float(np.mean([np.mean(a) for a in rec.values()])) for g,rec in groups.items()}
             for m,groups in by_method.items()}
    aggregate, comparisons = {}, {}
    for m,g in means.items():
        values = [g[k] for k in sorted(g)]
        aggregate[m] = dict(R1=float(np.mean(values)), ci95=group_ci(values,SEED,5000), songs=len(values))
        if m != actual and actual in means:
            groups = sorted(g.keys() & means[actual].keys())
            delta = [means[actual][k]-g[k] for k in groups]
            comparisons[m] = dict(delta=float(np.mean(delta)), ci95=group_ci(delta,SEED,5000), songs=len(delta))
    return dict(tasks=len(tasks), methods=aggregate, paired_differences=comparisons)


def phase_scores(net, sources, audio, tokens):
    result = []
    for source in sources:
        uid = source["source_uid"]
        key = source["audio_sha256"]
        offset = source["audio_offset_frames"]
        motion, dv = tokens[uid]
        music, mv, signal, zero_music = audio[key]
        alternatives = [v for k,v in sorted(audio.items()) if k!=key]
        # Donor fixed by identity, before seeing model scores.
        swap, sv, wrong_signal, _ = alternatives[0]
        lo,hi = source["shared_source_interval"]
        phases = []
        for p in source["phases"]:
            t = p["frame"]
            a,b = p["evidence_span"]
            if not (lo <= a <= t < b <= hi and RADIUS <= t < len(dv)+RADIUS and dv[t-RADIUS]):
                continue
            q = t+offset+np.array(OFFSETS)
            if (q.min() >= RADIUS and q.max() <= min(len(mv),len(sv))+RADIUS-1
                    and q.max() <= min(len(signal["onset"]),len(wrong_signal["onset"]))-1
                    and interpolate(music,mv,q,RADIUS) is not None
                    and interpolate(swap,sv,q,RADIUS) is not None):
                phases.append(p)
        if not phases:
            continue
        t = np.array([p["frame"] for p in phases])
        kinds = [p["type"] for p in phases]
        d = motion[t-RADIUS]
        scores = {name:[] for name in ("Content","Duration","Energy","Rule","MusicSwap","BothZero")}
        for delta in OFFSETS:
            q = t+offset+delta
            scores["Content"].append(cosine(interpolate(music,mv,q,RADIUS),d))
            scores["MusicSwap"].append(cosine(interpolate(swap,sv,q,RADIUS),d))
            scores["BothZero"].append(0.)
            scores["Energy"].append(stage_score(signal,t+offset,kinds,delta,"energy"))
            scores["Rule"].append(stage_score(signal,t+offset,kinds,delta))
            scores["Duration"].append(0.)
        result.append(dict(source=uid, recording=source["raw_motion_sha256"], audio=key,
                           phase_frames=t.tolist(), positive=OFFSETS.index(0), scores=scores))
    return result


def event_banks(rows, sources, audio, tokens):
    """A common crop length; use known physical-time positives and disjoint negatives."""
    sources = {s["source_uid"]:s for s in sources}
    prepared = []
    for row in rows:
        uid = row["source_uid"]
        s = sources[uid]
        a,b = row["source_interval"]
        if b-a < 90:
            continue
        start = (a+b-90)//2
        t = np.arange(start,start+90)
        d,dv = tokens[uid]
        d = interpolate(d,dv,t,RADIUS)
        key = row["audio_sha256"]
        q = t+s["audio_offset_frames"]
        m,mv,sig,_ = audio[key]
        q0,q1 = float(q[0]),float(q[-1]+1)
        mvectors = interpolate(m,mv,q,RADIUS)
        if d is None or mvectors is None or q0<0 or q1>len(sig["energy"]):
            continue
        prepared.append(dict(row=row,d=d,m=mvectors,q=q,span=[q0,q1]))
    result = defaultdict(list)
    for p in prepared:
        row = p["row"]
        key = row["audio_sha256"]
        # Self-recorded physical positive. All negatives come from other recordings.
        same, hard = [], []
        for n in prepared:
            r = n["row"]
            if r["raw_motion_sha256"] == row["raw_motion_sha256"]:
                continue
            same_song = r["audio_sha256"] == key
            disjoint = max(n["span"][0],p["span"][0])-min(n["span"][1],p["span"][1])>=15
            if same_song and not disjoint:
                continue
            mismatch = abs(r["motion_activity"]-row["motion_activity"])
            if same_song:
                same.append((mismatch,r["event_uid"],n))
            if mismatch <= max(.2,.2*row["motion_activity"]):
                hard.append((mismatch,r["event_uid"],n))
        m,mv,sig,_ = audio[key]
        swap,sv,_,_ = next(v for k,v in sorted(audio.items()) if k!=key)
        swapped = interpolate(swap,sv,p["q"],RADIUS)
        if swapped is None:
            continue
        levels = [float(np.mean(v[2]["energy"])) for v in audio.values()]
        music_activity = float(np.mean(np.interp(p["q"],np.arange(len(sig["energy"])),sig["energy"])))
        for name,pool in (("same_song_wrong_time",same),("equal_duration_near_motion_energy",hard)):
            if len(pool)<4:
                continue
            bank = [p]+[n for _,_,n in sorted(pool,key=lambda z:z[:2])[:4]]
            ids = [n["row"]["event_uid"] for n in bank]
            scores = dict(Content=[cosine(p["m"],n["d"]) for n in bank],
                          MusicSwap=[cosine(swapped,n["d"]) for n in bank],
                          Energy=[-float(abs(music_activity-n["row"]["motion_activity"]/(1+n["row"]["motion_activity"]))) for n in bank],
                          Duration=[0.]*5,BothZero=[0.]*5)
            result[name].append(dict(audio=key,recording=row["raw_motion_sha256"],query=row["event_uid"],
                                     candidate_events=ids,positive=0,scores=scores))
    return result


def run(out):
    if not read(PREVIEW_ROOT/"browser_receipt.json")["passed"]:
        raise ValueError("Watch main effect before comparative scoring")
    out = Path(out)
    if out.exists():
        raise ValueError("Fresh output required")
    net = Frozen()
    sources = [s for s in net.manifest["sources"] if s["role"]=="val"]
    train = [s for s in net.manifest["sources"] if s["role"]=="train"]
    for field in ("source_group","audio_sha256","raw_motion_sha256","recording_uid"):
        if {s[field] for s in sources}&{s[field] for s in train}:
            raise ValueError(f"Roles overlap through {field}")
    audio, motion = {}, {}
    for s in sources:
        key = s["audio_sha256"]
        if key not in audio:
            path = checked(dict(path=str(AF/(key+".npz")),sha256=s["features_sha256"]))
            with np.load(path,allow_pickle=False) as features:
                m,mv = net.music_tokens(features["values"],features["mask"])
            audio[key] = (m,mv,audio_signals(checked(s["audio"])),None)
        with np.load(checked(s["source_arrays"]),allow_pickle=False) as data:
            values,mask = net.motion(data["R"])
        motion[s["source_uid"]] = net.motion_tokens(values,mask)
    phases = phase_scores(net,sources,audio,motion)
    events = event_banks([r for r in net.manifest["rows"] if r["role"]=="val"],sources,audio,motion)
    summary = {"phase_offsets":summaries(phases), **{k:summaries(v) for k,v in events.items()}}
    write(out/"result.json",dict(model=net.binding(), split="HISTORICALLY_EXPOSED_VAL",
          sources=len(sources), songs=len(audio), phase=phases, events=dict(events), summary=summary,
          registered_blind_test=dict(status="NO_VERIFIED_UNEXPOSED_MUSIC_MANIFEST", evaluated=0,
              requirements=["new music work and PCM hashes","source and recording isolation",
                            "physical clock evidence","no prior training, tuning or checkpoint selection exposure"],
              offsets=list(OFFSETS), pairwise_group_ci_lower_bound=0),
          can_route=False, training_admitted=False))
    print("R49_VAL",len(audio),len(phases),{k:len(v) for k,v in events.items()},flush=True)

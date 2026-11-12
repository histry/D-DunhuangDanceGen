"""Read-only numerical event sidecars and independent human-anchor contracts.
Detected peaks are kinematic/acoustic proposals, never event correspondence truth.
"""
import numpy as np
from scipy.signal import find_peaks

PARTS={"upper":(3,6,9,12,13,14,15,16,17,18,19,20,21),
       "lower":(1,2,4,5,7,8,10,11),"root":(0,)}
TYPES=("start","speed_peak","deceleration","turn","hold")
EXTRACTOR=dict(schema="numerical_event_sidecar_v1",fps=30,min_peak_distance_frames=6,
    prominence_mad=1.,hold_quantile=.2,confidence="prominence/(prominence+robust_scale)",
    contacts="UNKNOWN",force_or_intent="UNKNOWN",correspondence_validated=False,
    context="whole source before crop; selected events sorted by physical time",selection="equal per-event-type quota, confidence backfill, cap then physical ordering")

def peaks(signal,kind,part):
    signal=np.asarray(signal,float)
    if signal.ndim!=1 or len(signal)<3 or not np.isfinite(signal).all(): raise ValueError("Finite continuous signal required")
    scale=max(float(np.median(np.abs(signal-np.median(signal)))*1.4826),1e-6)
    ids,props=find_peaks(signal,distance=6,prominence=scale)
    return [dict(frame=int(i),seconds=float(i/30),event_type=kind,body_part=part,
        confidence=float(p/(p+scale)),validated=False) for i,p in zip(ids,props["prominences"])]

def motion_events(features):
    x=np.asarray(features); speed=np.linalg.norm(x[:,:66].reshape(-1,22,3),axis=-1)
    events=[]
    for part,joints in PARTS.items():
        s=speed[:,joints].mean(-1)
        events+=peaks(s,"speed_peak",part)
        growth=np.maximum(np.diff(s,prepend=s[0])*30,0)
        events+=peaks(growth,"start",part)
        events+=peaks(np.maximum(-np.diff(s,prepend=s[0])*30,0),"deceleration",part)
        # A hold proposal needs a run of low speed, rather than a single local minimum.
        low=s<=np.quantile(s,.2);edges=np.diff(np.r_[False,low,False].astype(int));a=np.flatnonzero(edges==1);b=np.flatnonzero(edges==-1)
        for lo,hi in zip(a,b):
            if hi-lo>=6:
                i=(int(lo)+int(hi)-1)//2
                events.append(dict(frame=i,seconds=i/30,event_type="hold",body_part=part,confidence=.5,validated=False))
    events+=peaks(np.abs(x[:,1]),"turn","root") # native Y-up root angular velocity component
    return sorted(events,key=lambda r:(r["frame"],r["body_part"],r["event_type"]))

def music_events(values,mask):
    # Positive spectral flux from fixed log-mel channels. No tempo/bar/contact labels.
    x=np.asarray(values); valid=np.asarray(mask)[:,18:50].all(-1)
    flux=np.maximum(np.diff(x[:,18:50],axis=0,prepend=x[:1,18:50]),0).mean(-1)
    out=peaks(flux,"onset","music")
    control_onsets=peaks(x[:,1],"onset","music")
    for e in out:
        e["detector_agreement"]=any(abs(e["frame"]-p["frame"])<=3 for p in control_onsets)
    accents=peaks(x[:,11],"accent","music")
    beats=peaks(x[:,2],"beat","music")
    for e in accents+beats:
        e["detector_agreement"]=any(abs(e["frame"]-p["frame"])<=3 for p in out)
    out+=accents+beats
    low=x[:,0]<=np.quantile(x[:,0],.1)
    edges=np.diff(np.r_[False,low,False].astype(int));a=np.flatnonzero(edges==1);b=np.flatnonzero(edges==-1)
    for lo,hi in zip(a,b):
        if hi-lo>=6:
            frame=(int(lo)+int(hi)-1)//2
            out.append(dict(frame=frame,seconds=frame/30,event_type="silence",body_part="music",
                confidence=.5,detector_agreement=False,validated=False))
    return sorted([r for r in out if valid[r["frame"]]],key=lambda e:(e["frame"],e["event_type"]))

def crop_events(events,start,frames=120,cap=24):
    selected=[dict(e,relative_seconds=(e["frame"]-start)/30) for e in events if start<=e["frame"]<start+frames]
    # Detector confidence only; no matching score participates in this selection.
    rank=lambda e:(-e["confidence"],e["frame"],e["body_part"],e["event_type"])
    kinds=sorted({e["event_type"] for e in selected});kept=[]
    # Equal per-type quota before confidence backfill prevents derivative events crowding holds/peaks.
    quota=max(1,cap//max(len(kinds),1))
    for kind in kinds:kept.extend(sorted([e for e in selected if e["event_type"]==kind],key=rank)[:quota])
    if len(kept)<cap:
        identities={(e["frame"],e["body_part"],e["event_type"]) for e in kept}
        remaining=[e for e in selected if (e["frame"],e["body_part"],e["event_type"]) not in identities]
        kept.extend(sorted(remaining,key=rank)[:cap-len(kept)])
    return sorted(kept[:cap],key=lambda e:(e["frame"],e["body_part"],e["event_type"]))

def review_packet(row):
    return dict(schema="independent_event_anchors_v1",window_uid=row["window_uid"],role=row["role"],
        source_group=row["source_group"],recording_uid=row["recording_uid"],
        audio_sha256=row["audio"]["sha256"],motion_sha256=row["raw_motion"]["sha256"],
        source_interval_seconds=[(row["start_frame"]+15)/30,(row["start_frame"]+135)/30],
        audio_offset_seconds=row["audio_offset_seconds"],validated=False,reviewers=[],anchors=[],
        label_origin="UNREVIEWED",model_predictions_shown=False,
        instructions="Review original audio/video/motion independently. Mark type, body part, ambiguity and explicit NULL. Do not copy detector/OT proposals.")

def validate_anchors(packet,row):
    for key in ("window_uid","role","source_group","recording_uid"):
        if packet.get(key)!=row[key]: raise ValueError("Annotation source identity differs")
    if packet.get("audio_sha256")!=row["audio"]["sha256"] or packet.get("motion_sha256")!=row["raw_motion"]["sha256"]:
        raise ValueError("Annotation source hash differs")
    interval=[(row["start_frame"]+15)/30,(row["start_frame"]+135)/30]
    if packet.get("source_interval_seconds")!=interval or packet.get("audio_offset_seconds")!=row["audio_offset_seconds"]:
        raise ValueError("Annotation physical clock differs")
    if packet.get("schema")!="independent_event_anchors_v1" or packet.get("validated") is not True:
        raise ValueError("Independent event anchors are not validated")
    reviewers=packet.get("reviewers",[])
    if len(set(reviewers))<2 or any(not isinstance(x,str) or not x.strip() for x in reviewers):
        raise ValueError("Two distinct independent reviewers required")
    if packet.get("label_origin")!="independent_source_review" or packet.get("model_predictions_shown") is not False:
        raise ValueError("Model-derived labels cannot validate correspondence")
    for a in packet["anchors"]:
        if a.get("agreement") is not True or set(a.get("reviewers",[]))!=set(reviewers): raise ValueError("Anchor agreement missing")
        if a.get("event_type") not in TYPES or a.get("body_part") not in PARTS: raise ValueError("Unknown typed body event")
        t=a.get("music_seconds");d=a.get("motion_seconds");null=a.get("is_null")
        if type(t) not in (int,float) or not np.isfinite(t) or not interval[0]+row["audio_offset_seconds"]<=t<interval[1]+row["audio_offset_seconds"]:
            raise ValueError("Music anchor outside observed physical support")
        if type(null) is not bool or (null and d is not None): raise ValueError("Explicit NULL semantics required")
        if not null and (type(d) not in (int,float) or not np.isfinite(d) or not interval[0]<=d<interval[1]):
            raise ValueError("Motion anchor outside observed physical support")
        if type(a.get("ambiguous")) is not bool: raise ValueError("Ambiguity must be reviewed")
    return packet["anchors"]

def train_lags(packets,rows,bound=.5,min_recordings=3):
    buckets={}
    for p in packets:
        r=rows[p["window_uid"]]; anchors=validate_anchors(p,r)
        if r["role"]!="train": raise ValueError("Only Train anchors may calibrate lags")
        for a in anchors:
            if not a["is_null"] and not a["ambiguous"]:
                key=(a["event_type"],a["body_part"])
                buckets.setdefault(key,[]).append((a["motion_seconds"]-(a["music_seconds"]-r["audio_offset_seconds"]),r["recording_uid"]))
    result={}
    for key,items in buckets.items():
        calibrated=len({r for _,r in items})>=min_recordings
        lag=float(np.clip(np.median([x for x,_ in items]),-bound,bound)) if calibrated else 0.
        result["/".join(key)]=dict(seconds=lag,calibrated=calibrated,recordings=len({r for _,r in items}))
    return dict(default_seconds=0.,default_calibrated=False,bound_seconds=bound,types=result)

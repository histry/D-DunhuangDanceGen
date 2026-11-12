"""Source-only key-motion / boundary contract. No generated boundary or Native admission."""
import hashlib,json
import numpy as np
from dunhuang.io import checked
from .event_plan import validate
from .time_map import source_positions

SCHEMA="event_primitive_condition_v2"

def build(plan,observed,beat_frames=(),radius=12):
    validate(plan,plan["audio_sha256"],plan["frames"])
    if radius!=12:raise ValueError("Registered primitive radius is twelve")
    events=[]
    for e in plan["events"]:
        data=np.load(checked(e["reference"]),allow_pickle=False)
        mask=np.load(checked(observed[e["source_uid"]]),allow_pickle=False)
        if mask.shape!=data.shape or mask.dtype!=bool:raise ValueError("Source observation mask differs")
        a,b=e["music_interval"];lo,hi=e["source_interval"]
        candidates=[e["primitive"]["music_frame"]]+[int(t) for t in beat_frames if a+radius<=t<b-radius]
        # Up to three source-only keys; no claim of learned music interpretation.
        ticks=sorted(set(candidates))
        if len(ticks)>3:
            center=e["primitive"]["music_frame"]
            beats=[t for t in ticks if t!=center]
            ticks=sorted(set([center,beats[0],beats[-1]]))
        keys=[]
        for tick in ticks:
            source=float(source_positions(e,np.array([tick]))[0])
            left=int(np.floor(source));right=int(np.ceil(source))
            joints=[j for j in e["primitive"]["joints"] if mask[left,7+6*j:13+6*j].all() and mask[right,7+6*j:13+6*j].all()]
            if not joints:continue
            keys.append(dict(source_frame=source,music_frame=tick,observed_joints=joints,confidence_scope="SOURCE_ONLY"))
        events.append(dict(event_uid=e["event_uid"],source_uid=e["source_uid"],reference=e["reference"],
            source_interval=e["source_interval"],music_interval=e["music_interval"],time_warp=e["time_warp"],
            source_observed=observed[e["source_uid"]],key_motion=keys,
            boundary=dict(left_source_interval=[lo,min(hi,lo+13)],right_source_interval=[max(lo,hi-13),hi],
                scope="observed source boundaries, NOT generated neighboring context"),
            root_contact=dict(root_observed=bool(mask[lo:hi,4:13].all()),contact_semantics="UNKNOWN"),
            expression=dict(mode="identity_source_reference",learned=False),qualification="SHADOW"))
    value=dict(schema=SCHEMA,fps=30,frames=plan["frames"],audio_sha256=plan["audio_sha256"],
        time_map_schema="event_time_warp_motion74_v1",events=events,can_route=False)
    value["semantic_digest"]=identity(value);return value

def identity(contract):
    payload=dict(schema=contract["schema"],frames=contract["frames"],audio_sha256=contract["audio_sha256"],events=[])
    for e in contract["events"]:
        payload["events"].append({k:e[k] for k in ("event_uid","source_uid","source_interval","music_interval","time_warp","key_motion","boundary","root_contact","expression")})
        payload["events"][-1]["reference_sha256"]=e["reference"]["sha256"]
        payload["events"][-1]["observed_sha256"]=e["source_observed"]["sha256"]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def verify(contract,plan):
    validate(plan,plan["audio_sha256"],plan["frames"])
    if contract.get("fps")!=30 or contract.get("time_map_schema")!="event_time_warp_motion74_v1":raise ValueError("Primitive clock schema differs")
    if contract["schema"]!=SCHEMA or contract["can_route"] is not False or contract["frames"]!=plan["frames"] or contract["audio_sha256"]!=plan["audio_sha256"]:raise ValueError("Primitive condition identity differs")
    by_id={e["event_uid"]:e for e in plan["events"]}
    if len(contract["events"])!=len(by_id):raise ValueError("Primitive contract event population differs")
    for e in contract["events"]:
        p=by_id[e["event_uid"]]
        for key in ("source_uid","source_interval","music_interval","time_warp","reference"):
            if e[key]!=p[key]:raise ValueError("Primitive contract canonical map differs")
        data=np.load(checked(e["reference"]),allow_pickle=False)
        mask=np.load(checked(e["source_observed"]),allow_pickle=False)
        if mask.shape!=data.shape or mask.dtype!=bool:raise ValueError("Primitive observation mask differs")
        lo,hi=e["source_interval"];a,b=e["music_interval"]
        expected=dict(left_source_interval=[lo,min(hi,lo+13)],right_source_interval=[max(lo,hi-13),hi],
            scope="observed source boundaries, NOT generated neighboring context")
        if e["boundary"]!=expected or e["root_contact"]["root_observed"]!=bool(mask[lo:hi,4:13].all()):raise ValueError("Primitive boundary evidence differs")
        if e["expression"]!=dict(mode="identity_source_reference",learned=False):raise ValueError("Untrained expression must stay identity")
        if len(e["key_motion"])>3:raise ValueError("Registered key-motion budget differs")
        ticks=[k["music_frame"] for k in e["key_motion"]]
        if ticks!=sorted(set(ticks)):raise ValueError("Key-motion ordering differs")
        for key in e["key_motion"]:
            tick=key["music_frame"]
            if type(tick) is not int or not a+12<=tick<b-12:raise ValueError("Primitive context outside interval")
            source=float(source_positions(p,np.array([tick]))[0])
            left=int(np.floor(source));right=int(np.ceil(source))
            joints=key["observed_joints"]
            if not joints or len(joints)!=len(set(joints)) or any(j not in p["primitive"]["joints"] or not mask[left,7+6*j:13+6*j].all() or not mask[right,7+6*j:13+6*j].all() for j in joints):raise ValueError("Unobserved key-motion joint")
            if key["confidence_scope"]!="SOURCE_ONLY":raise ValueError("Unknown key-motion confidence scope")
            if abs(source-key["source_frame"])>1e-6:raise ValueError("Primitive clock differs")
        if e["qualification"]!="SHADOW" or e["root_contact"]["contact_semantics"]!="UNKNOWN":raise ValueError("Primitive cannot pre-admit physical quality")
    if identity(contract)!=contract["semantic_digest"]:raise ValueError("Primitive contract digest differs")
    return contract

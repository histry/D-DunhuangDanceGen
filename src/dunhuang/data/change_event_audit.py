"""E0/E1 on frozen event boundaries, identical music slots and unchanged scaling."""
from collections import Counter
import numpy as np
from dunhuang.contracts import MotionAdapter
from dunhuang.planning.time_map import source_positions
from .change_event_phases import UPPER,phases_for_core,choose_anchor,choose_keys
from .split_registry import stable

def normalized_support(native,frame,a,b,scale,bias,observed_joints):
    if frame is None or not a+12<=frame<b-12 or not set(UPPER)<=set(observed_joints):return False
    clip=MotionAdapter.to_edge(native[frame-12:frame+13].copy())[:,79:127]
    normalized=clip*scale[79:127]+bias[79:127]
    return bool(np.isfinite(normalized).all() and (abs(normalized)<=1+32*np.finfo(np.float32).eps).all())

def audit_event(old,native,profiles,phases,observed_joints,scale,bias,catalog_sha,source_record):
    a,b=old["source_interval"];length=b-a
    if not 0<=a<b<=len(native) or length<25:raise ValueError("Frozen Core outside actual source")
    old_peak=a+1+int(profiles["old_upper10"][a+1:b].argmax())
    legacy_center=a+int(np.clip(old_peak-a,max(13,round(.35*length)),min(length-14,round(.65*length))))
    if legacy_center!=old["center"]:raise ValueError("Original V1 center cannot be reproduced")
    old_support=normalized_support(native,legacy_center,a,b,scale,bias,observed_joints)
    if old_support!=(old["feasibility"]["temporal8"]=="PASS"):raise ValueError("Original V1 normalization support cannot be reproduced")
    controlled=a+1+int(profiles["upper8"][a+1:b].argmax())
    event_phases=phases_for_core(phases,a,b,controlled)
    anchor=choose_anchor(event_phases,controlled,a,b,observed_joints,profiles["upper8"])
    keys=choose_keys(event_phases,anchor,a,b);new_center=None if anchor is None else anchor["frame"]
    new_support=normalized_support(native,new_center,a,b,scale,bias,observed_joints)
    boundary=dict(left=[a-12,a] if a>=12 else None,right=[b,b+12] if b+12<=len(native) else None,scope="OBSERVED_SOURCE_ONLY")
    e=dict(schema="change_event_v2_shadow",event_uid=old["event_uid"],source_uid=old["source_uid"],recording_uid=old["recording_uid"],
        source_sha256=old["source_sha256"],source_interval=[a,b],duration_frames=length,reference=old["reference"],
        original_peak=old_peak,controlled_peak=controlled,safe_anchor=anchor,anchor_status="SAFE_SOURCE_ANCHOR" if anchor else "NO_SAFE_ANCHOR",
        kinematic_phases=event_phases,key_motion=keys,boundary_context=boundary,discovery_read_support=[0,len(native)],
        observed_joints=observed_joints,controlled_joints=list(UPPER),normalization_support=new_support,
        primitive_evidence=dict(real_source_context=anchor is not None,maximum_not_relocated=True,contact_semantics="UNKNOWN",root_is_controlled=False),
        segmentation_evidence=dict(method="FROZEN_V1_CORE_UNCHANGED",source_catalog_sha256=catalog_sha,full_record_discovery_dependency=source_record["discovery_dependency"]),
        music_semantic_label="UNKNOWN",human_reviewed=False,native_qualified=False,can_route=False)
    e["variant_uid"]="change-e1:"+stable(dict(source=old["source_sha256"],event=old["event_uid"],core=[a,b],anchor=new_center,
        phases=[(p["frame"],p["type"]) for p in keys],joints=list(UPPER),algorithm="whole_record_upper8_e1_v1"))
    diagnostic=dict(event_uid=old["event_uid"],source_uid=old["source_uid"],recording_uid=old["recording_uid"],frames=length,
        original_peak=old_peak,controlled_peak=controlled,legacy_center=legacy_center,new_center=new_center,
        old_peak_clamped=legacy_center!=old_peak,joint_set_peak_changed=old_peak!=controlled,
        old_center_shift_frames=legacy_center-old_peak,controlled_peak_shift_frames=controlled-old_peak,
        e0_captures_controlled_peak=abs(legacy_center-controlled)<=12,
        e1_captures_controlled_peak=new_center is not None and abs(new_center-controlled)<=12,
        e0_source_context=True,e1_source_context=anchor is not None,e0_normalized_support=old_support,e1_normalized_support=new_support,
        lower_phase_speed_mean=float(profiles["lower"][a:b].mean()),
        boundary_upper_active=bool(profiles["upper8"][a]>.2 or profiles["upper8"][b-1]>.2),
        phases_with_boundary_truncation=sum(p["evidence_truncated_by_core"] for p in event_phases if "evidence_truncated_by_core" in p),
        left_boundary_context=boundary["left"] is not None,right_boundary_context=boundary["right"] is not None)
    return e,diagnostic

def slot_eligibility(entry,slot,beats,mode="temporal"):
    a,b=entry["source_interval"];center=entry.get("center")
    if center is None:return False,"NO_SAFE_ANCHOR",None
    if mode=="temporal" and not entry["normalization_support"]:return False,"NORMALIZER_SUPPORT",None
    start,end=slot["start"],slot["end"];ratio=(end-start)/(b-a)
    if not .75<=ratio<=1.33:return False,"DURATION_RATIO",None
    tick=start+round((center-a)*ratio);near=[t for t in beats if abs(t-tick)<=4]
    if near:tick=min(near,key=lambda t:abs(t-tick))
    if tick-12<start or tick+12>=end:return False,"MUSIC_CONTEXT",None
    warp=np.array([[a,start],[center-12,tick-12],[center+12,tick+12],[b,end]],float)
    if np.any(np.diff(warp,axis=0)<=0):return False,"ZERO_OR_NONMONOTONE_WARP",None
    rates=np.diff(warp[:,1])/np.diff(warp[:,0])
    if not np.isfinite(rates).all() or (rates<.75).any() or (rates>1.33).any():return False,"LOCAL_RATE_LIMIT",None
    event=dict(source_interval=[a,b],music_interval=[start,end],time_warp=warp.tolist())
    source_positions(event,np.arange(start,end))
    return True,"PASS",warp.tolist()

def summarize(diagnostics,matrices,slot_records):
    n=len(diagnostics);result=dict(events=n,sources=len({r["source_uid"] for r in diagnostics}),recordings=len({r["recording_uid"] for r in diagnostics}),
        old_peak_clamped=sum(r["old_peak_clamped"] for r in diagnostics),joint_set_peak_changed=sum(r["joint_set_peak_changed"] for r in diagnostics),
        old_center_abs_shift_mean_frames=float(np.mean([abs(r["old_center_shift_frames"]) for r in diagnostics])),
        controlled_peak_abs_shift_mean_frames=float(np.mean([abs(r["controlled_peak_shift_frames"]) for r in diagnostics])),
        e0=dict(source_safe=sum(r["e0_source_context"] for r in diagnostics),controlled_peak_captured=sum(r["e0_captures_controlled_peak"] for r in diagnostics),normalized_support=sum(r["e0_normalized_support"] for r in diagnostics)),
        e1=dict(source_safe=sum(r["e1_source_context"] for r in diagnostics),controlled_peak_captured=sum(r["e1_captures_controlled_peak"] for r in diagnostics),normalized_support=sum(r["e1_normalized_support"] for r in diagnostics)),
        boundary_proxy=dict(active_boundaries=sum(r["boundary_upper_active"] for r in diagnostics),events_with_truncated_phase_span=sum(r["phases_with_boundary_truncation"]>0 for r in diagnostics)),
        real_music_slot_count=len(slot_records),planner={},human_event_truth=False,optimizer_calls=0,operational_qualified=False)
    for key,m in matrices.items():
        counts=m.sum(-1);covered=[i for i in range(n) if m[:,i].any()] if len(m) else []
        result["planner"][key]=dict(legal_candidate_pairs=int(m.sum()),mean_candidates_per_slot=float(counts.mean()) if len(counts) else None,
            empty_slots=int((counts==0).sum()),events_ever_eligible=len(covered),sources_ever_eligible=len({diagnostics[i]["source_uid"] for i in covered}))
    result["e2_boundary_resegmentation_started"]=False
    result["e3_generation_started"]=False
    result["e1_controlled_peak_coverage_improved"]=result["e1"]["controlled_peak_captured"]>result["e0"]["controlled_peak_captured"]
    result["duration_histogram"]={str(k):int(v) for k,v in sorted(Counter(r["frames"] for r in diagnostics).items())}
    return result

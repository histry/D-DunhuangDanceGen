"""E1-W: same raw sources, centers, slots and bounds; change only map solving."""
import argparse,collections,json,subprocess
from pathlib import Path
from fractions import Fraction
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.data.change_event_audit import slot_eligibility
from dunhuang.planning.feasible_time_map import solve
from dunhuang.planning.time_map import source_positions
from dunhuang.research.data_event_v3.study import run as source_audit

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def oracle(entry,slot):
    """Independent exhaustive integer-frame oracle, not interval algebra."""
    a,b=entry["source_interval"];A,B=slot["start"],slot["end"];c=entry["center"]
    if c is None or not entry["normalization_support"]:return np.array([],np.int64)
    if c-12<a or c+12>=b or not 75*(b-a)<=100*(B-A)<=133*(b-a):return np.array([],np.int64)
    ticks=np.arange(A+12,B-12,dtype=np.int64)
    ds=np.array([c-12-a,24,b-c-12])
    if np.any(ds<=0):return np.array([],np.int64)
    dt=np.stack([ticks-12-A,np.full_like(ticks,24),B-ticks-12],axis=1)
    return ticks[np.all((dt>0)&(100*dt>=75*ds)&(100*dt<=133*ds),axis=1)]

def stats(values):
    values=np.asarray(values,float)
    return dict(count=len(values),mean=float(values.mean()) if len(values) else None,
        median=float(np.median(values)) if len(values) else None,
        p95=float(np.quantile(values,.95)) if len(values) else None,max=float(values.max()) if len(values) else None)

def summarize(good,ticks,records,old,new,diagnostics,timelines):
    sources=sorted({e["source_uid"] for e in old});summary={}
    for arm,m in good.items():
        index=int(arm[1]);is_e1=arm in ("T1","T3")
        capture=np.array([d["e1_captures_controlled_peak"] if is_e1 else d["e0_captures_controlled_peak"] for d in diagnostics],bool)
        centers=[e["safe_anchor"]["frame"] if e["safe_anchor"] else None for e in new] if is_e1 else [e["center"] for e in old]
        distance=[];shifts=[];hit=0
        for i,r in enumerate(records):
            slot=r["slot"];A,B=slot["start"],slot["end"];beats=[t for t in timelines[r["timeline_identity"]]["beats"] if A<=t<B]
            for j in np.flatnonzero(m[i]):
                t=int(ticks[index,i,j]);a,b=old[j]["source_interval"]
                preferred=A+round(Fraction((centers[j]-a)*(B-A),b-a));shifts.append(abs(t-preferred))
                if beats:
                    d=min(abs(t-v) for v in beats);distance.append(d);hit+=d==0
        active=m.any(0)
        summary[arm]=dict(legal_candidate_pairs=int(m.sum()),empty_slots=int((m.sum(1)==0).sum()),
            events_ever_eligible=int(active.sum()),sources_ever_eligible=len({old[j]["source_uid"] for j in np.flatnonzero(active)}),
            peak_capturing_eligible_events=int((active&capture).sum()),peak_capturing_candidate_pairs=int(m[:,capture].sum()),
            exact_slot_beat_pairs=int(hit),beat_distance_frames=stats(distance),linear_displacement_frames=stats(shifts),
            per_music_frames={str(frames):int(m[[i for i,r in enumerate(records) if r["frames"]==frames]].sum()) for frames in sorted({r["frames"] for r in records})},
            per_recording=[dict(recording_index=k+1,events=sum(e["source_uid"]==uid for e in old),
                legal_pairs=int(m[:,[e["source_uid"]==uid for e in old]].sum())) for k,uid in enumerate(sources)])
    changes={}
    for before,after in (("T0","T2"),("T1","T3")):
        common=good[before]&good[after];i,j=np.where(common);delta=abs(ticks[int(after[1]),i,j]-ticks[int(before[1]),i,j])
        beat_delta=[]
        for row,col in zip(i,j):
            rec=records[row];A,B=rec["slot"]["start"],rec["slot"]["end"]
            beats=[t for t in timelines[rec["timeline_identity"]]["beats"] if A<=t<B]
            if beats:
                old_distance=min(abs(int(ticks[int(before[1]),row,col])-t) for t in beats)
                new_distance=min(abs(int(ticks[int(after[1]),row,col])-t) for t in beats)
                beat_delta.append(new_distance-old_distance)
        if any(v>0 for v in beat_delta):raise AssertionError("Beat-optimal solver worsened a common feasible map")
        changes[before+"_to_"+after]=dict(common_beat_distance_improved=sum(v<0 for v in beat_delta),
            common_beat_distance_unchanged=sum(v==0 for v in beat_delta),common_beat_distance_worsened=sum(v>0 for v in beat_delta),
            recovered_pairs=int((good[after]&~good[before]).sum()),
            lost_pairs=int((good[before]&~good[after]).sum()),common_pairs=int(common.sum()),
            changed_ticks_on_common_pairs=int((delta!=0).sum()),absolute_tick_change_frames=stats(delta))
    changes["center_conflict"]=dict(old_center_only_under_legacy=int((good["T0"]&~good["T1"]).sum()),
        recovered_old_center_only_by_T3=int((good["T0"]&~good["T1"]&good["T3"]).sum()),
        T0_pairs_still_unavailable_to_T3=int((good["T0"]&~good["T3"]).sum()),
        T3_pairs_not_in_T0=int((good["T3"]&~good["T0"]).sum()),
        net_T0_minus_T1=int(good["T0"].sum()-good["T1"].sum()))
    return summary,changes

def run(config,out,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    if config["arms"]!=["T0","T1","T2","T3"]:raise ValueError("Registered arm order differs")
    if config["optimizer_calls"] or config["statistics_fit_calls"] or config["allow_generation"] or config["radius"]!=12 or config["rate_limits"]!=[.75,1.33]:raise ValueError("Registered mathematical study budget differs")
    frozen=Path(config["source_study"])
    seal=read(frozen/"public_receipt.json");published=read("reports/data_event_v3_receipt.json")
    if seal!=published or seal["exit_status"]!=0:raise ValueError("Frozen Data V3 successful receipt differs")
    previous=read(frozen/"public_summary.json")
    if sha(frozen/"public_summary.json")!=seal["summary_sha256"]:raise ValueError("Frozen source results changed")
    for name,digest in previous["evidence"].items():
        if sha(frozen/name)!=digest:raise ValueError("Frozen Data V3 evidence differs")
    parent=read(frozen/"main/protocol.json")
    refs=[ref(frozen/"main/protocol.json"),ref(frozen/"public_receipt.json"),ref(frozen/"public_summary.json")]
    refs+=[t[k] for t in parent["timelines"] for k in ("timeline","plan")]
    for r in refs:checked(r)
    protocol=dict(schema="primitive_warp_study_v1",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config=config,upstream=refs,baseline_commit="685ad0bcf13b9e7be085e1df3d3d580040f068c5",
        g2_receipt=ref("reports/source_id_03_receipt.json"),source_anchors_moved=False,rate_bounds_relaxed=False,
        default_planning_changed=False,optimizer_calls=0,statistics_fit_calls=0,edge_forward_calls=0,native_calls=0)
    write(out/"protocol.json",protocol)
    source_audit(checked(parent["assets"]),checked(parent["population"]),parent["config"],out/"source_audit",frozen/"main")
    audited=out/"source_audit";old=read(checked(parent["catalog"]))["entries"];new=read(audited/"event_catalog_v2_shadow.json")["entries"]
    diagnostics=read(audited/"event_diagnostics.json");records=read(audited/"music_slots.json")
    timelines={t["semantic_identity"]:t["payload"] for t in parent["timelines"]}
    if len(old)!=config["expected_events"] or len(records)!=config["expected_slots"] or len({e["source_uid"] for e in old})!=config["expected_sources"]:raise ValueError("Registered population changed")
    shape=(4,len(records),len(old));ticks=np.full(shape,-1,np.int32);warps=np.full(shape+(4,2),-1,np.int32)
    lower=np.full(shape,-1,np.int32);upper=lower.copy();good={arm:np.zeros(shape[1:],bool) for arm in config["arms"]}
    reasons={arm:collections.Counter() for arm in good};certificates=[];checks=0;max_inverse=0.;stage_checks=0
    for i,r in enumerate(records):
        beats=timelines[r["timeline_identity"]]["beats"];slot=r["slot"]
        for j,(e0,e1,d) in enumerate(zip(old,new,diagnostics)):
            if e0["event_uid"]!=e1["event_uid"] or e0["source_interval"]!=e1["source_interval"]:raise ValueError("Event alignment changed")
            for k,arm in enumerate(config["arms"]):
                c=e0["center"] if arm in ("T0","T2") else None if e1["safe_anchor"] is None else e1["safe_anchor"]["frame"]
                support=d["e0_normalized_support"] if arm in ("T0","T2") else e1["normalization_support"]
                entry=dict(source_interval=e0["source_interval"],center=c,normalization_support=support)
                if arm in ("T0","T1"):
                    ok,reason,warp=slot_eligibility(entry,slot,beats)
                    t=int(warp[1][1]+12) if ok else None
                else:
                    solution=solve(entry,slot,beats);ok=solution["status"]=="PASS";reason=solution["status"];warp=solution["time_warp"];t=solution["tick"]
                    reference=oracle(entry,slot);checks+=1
                    bounds=solution["integer_domain"]
                    domain_ticks=np.arange(bounds[0],bounds[1]+1) if bounds else np.array([],np.int64)
                    if not np.array_equal(reference,domain_ticks):raise AssertionError("Exact solver domain differs from exhaustive feasible set")
                    if len(reference):
                        slot_beats=np.array([v for v in beats if slot["start"]<=v<slot["end"]],np.int64)
                        distances=abs(reference[:,None]-slot_beats).min(1) if len(slot_beats) else np.zeros(len(reference),np.int64)
                        a,b=e0["source_interval"];preferred=slot["start"]+round(Fraction((c-a)*(slot["end"]-slot["start"]),b-a))
                        winner=reference[np.lexsort((reference,abs(reference-preferred),distances))[0]]
                        if winner!=t:raise AssertionError("Selected tick is not the exhaustive beat/projection optimum")
                    if bounds:lower[k,i,j],upper[k,i,j]=bounds
                    certificates.append(dict(arm=arm,slot_index=i,event_uid=e0["event_uid"],certificate=solution))
                reasons[arm][reason]+=1
                if not ok:continue
                good[arm][i,j]=True;ticks[k,i,j]=t;warps[k,i,j]=warp
                phases=sorted({p["frame"] for p in e1["kinematic_phases"]})
                w=np.array(warp,float);times=np.interp(phases,w[:,0],w[:,1])
                if len(times)>1 and (np.diff(times)<=0).any():raise AssertionError("Mapped stage order inverted")
                back=source_positions(dict(source_interval=e0["source_interval"],music_interval=[slot["start"],slot["end"]],time_warp=warp),times)
                err=float(np.max(abs(back-phases))) if phases else 0.;max_inverse=max(max_inverse,err);stage_checks+=1
                if err>1e-7:raise AssertionError("Mapped stage clock is not invertible in observed source support")
        if i%10==0:print("WARP_SLOTS_VERIFIED",i+1,len(records),flush=True)
    with np.load(frozen/"main/planner_eligibility.npz") as z:
        if not np.array_equal(z["e0_temporal"],good["T0"]) or not np.array_equal(z["e1_temporal"],good["T1"]):raise ValueError("Legacy eligibility replay differs")
    for before,after in (("T0","T2"),("T1","T3")):
        if (good[before]&~good[after]).any():raise AssertionError("Complete solver lost legacy feasible pairs")
    arms,changes=summarize(good,ticks,records,old,new,diagnostics,timelines)
    result=dict(schema="primitive_warp_results_v1",events=len(old),sources=8,music_slots=len(records),arms=arms,changes=changes,
        rejection_reasons={k:dict(v) for k,v in reasons.items()},exhaustive_domain_checks=checks,
        stage_order_checks=stage_checks,stage_order_inversions=0,phase_inverse_max_error_frames=max_inverse,
        source_peak_coverage_unchanged=dict(E0=sum(d["e0_captures_controlled_peak"] for d in diagnostics),E1=sum(d["e1_captures_controlled_peak"] for d in diagnostics)),
        source_anchor_context_unchanged=True,rate_bounds_relaxed=False,default_planning_changed=False,
        gates=dict(E2_started=False,A2_new_split_candidates_rebuilt=False,A2_training_started=False,
            G2_previously_qualified=False,G2_training_started=False,G3_started=False,FineDance_admitted=False),
        optimizer_calls=0,statistics_fit_calls=0,edge_forward_calls=0,native_calls=0,operational_qualified=False)
    np.savez_compressed(out/"maps.npz",ticks=ticks,warps=warps,domain_lower=lower,domain_upper=upper,**good)
    write(out/"certificates.json",certificates);write(out/"result.json",result)
    if replay_from:
        original=Path(replay_from)
        if read(original/"result.json")!=result or read(original/"certificates.json")!=certificates:raise ValueError("Fresh warp readout/certificates differ")
        x=dict(np.load(original/"maps.npz"));y=dict(np.load(out/"maps.npz"))
        if set(x)!=set(y) or any(not np.array_equal(x[k],y[k]) for k in x):raise ValueError("Fresh warp arrays differ")
        write(out/"replay.json",dict(status="FRESH_RAW_SOURCE_WARP_REPLAY_VERIFIED",optimizer_calls=0,
            raw_source_decodes=8,event_ids=499,source_audit_replay=read(audited/"replay.json"),array_fields_verified=len(x),
            exhaustive_domain_checks=checks,stage_order_inversions=0))
    print("PRIMITIVE_WARP_COMPLETED",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.out,a.replay_from)
if __name__=="__main__":main()

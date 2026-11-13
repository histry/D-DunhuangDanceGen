"""Use E1-W in a Shadow call path; the default Planner is never changed."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.data.change_event_audit import slot_eligibility
from dunhuang.planning.feasible_time_map import solve
from dunhuang.planning.planner import plan as default_plan,_search
from dunhuang.planning.event_plan import candidate_identity,semantic_plan_id,validate
from dunhuang.planning.time_map import source_positions
from dunhuang.generation.soft_reference import build_native_event_reference,build_soft_reference,resample_native
from dunhuang.research.source_capacity.common import read,commit,fresh,same_arrays

def candidates(old,new,timing,arm):
    by_uid={e["event_uid"]:e for e in new};banks=[];all_uid=[]
    for slot in timing["slots"]:
        options=[]
        for entry in old:
            e1=by_uid[entry["event_uid"]]
            if entry["source_interval"]!=e1["source_interval"] or entry["reference"]["sha256"]!=e1["reference"]["sha256"]:raise ValueError("Shadow source or Core changed")
            center=entry["center"] if arm=="E0_legacy" else e1["safe_anchor"]["frame"] if e1["safe_anchor"] else None
            support=entry["feasibility"]["temporal8"]=="PASS" if arm=="E0_legacy" else e1["normalization_support"]
            record=dict(source_interval=entry["source_interval"],center=center,normalization_support=support)
            if arm=="E1_W":
                solution=solve(record,slot,timing["beats"]);ok=solution["status"]=="PASS";warp=solution["time_warp"];tick=solution["tick"]
            else:
                ok,reason,warp=slot_eligibility(record,slot,timing["beats"]);tick=int(warp[1][1]+12) if ok else None
            if not ok:continue
            warp=[[int(x),int(y)] for x,y in warp]
            event=copy.deepcopy(entry);a,b=entry["source_interval"]
            event.update(center=center,music_interval=[slot["start"],slot["end"]],time_warp=warp,
                primitive=dict(source_frame=center,music_frame=tick,joints=list(range(12,20)),radius=12),
                boundary_context=dict(left_source_frames=[a,min(a+4,b)],right_source_frames=[max(a,b-4),b],confidence="SOURCE_ONLY"),
                scores=dict(heuristic=-abs(entry["activity_percentile"]-slot["density"]),global_score=None,sync_score=None,preference=None),
                expression=dict(amplitude=1.,mode="identity_source_reference",learned_expression=False))
            event["candidate_uid"]=candidate_identity(event)
            options.append(event)
        banks.append(options);all_uid.append([e["candidate_uid"] for e in options])
    return banks,all_uid

def search(banks,top_k,beam_width):
    # Minimal research sequencing extension; parity is checked against the actual default.
    beams=[(0.,[])]
    for options in banks:
        if not options:return None
        ordered=sorted(options,key=lambda e:(-e["scores"]["heuristic"],e["event_uid"]))
        chosen=[next(e for e in ordered if e["source_uid"]==uid) for uid in dict.fromkeys(e["source_uid"] for e in ordered)]
        chosen=sorted(chosen,key=lambda e:-e["scores"]["heuristic"])[:top_k]
        expanded=[]
        for cost,sequence in beams:
            for event in chosen:
                transition=0.
                if sequence:transition=float(Rotation.from_matrix(np.swapaxes(np.array(sequence[-1]["exit"]),-1,-2)@np.array(event["entry"])).magnitude().mean())
                total=cost-event["scores"]["heuristic"]+.3*sum(e["event_uid"]==event["event_uid"] for e in sequence)+.04*sum(e["source_uid"]==event["source_uid"] for e in sequence)+.25*transition
                expanded.append((total,sequence+[event]))
        beams=sorted(expanded,key=lambda p:(p[0],[e["event_uid"] for e in p[1]]))[:beam_width]
    return beams[0]

def inspect_conditions(plan,entries,scale,bias):
    native,mask,phase,audit=build_native_event_reference(plan,plan["frames"])
    normalized,scaler_mask,_,soft=build_soft_reference(plan,plan["frames"],scale,bias)
    events={e["event_uid"]:e for e in entries};positions=[];checked_frames=0;phase_checks=0
    for event in plan["events"]:
        original=events[event["event_uid"]]
        if not set(range(12,20))<=set(original["observed_joints"]):raise ValueError("Upper8 source is not observed")
        a,b=event["music_interval"];ticks=np.arange(a,b);pos=source_positions(event,ticks);positions.extend(pos.tolist())
        sampled=resample_native(np.load(checked(event["reference"])),pos)
        for k,tick in enumerate(ticks):
            window=int(tick//75);local=int(tick-75*window)
            # Final short song tail may have no next full window.
            window=min(window,len(native)-1);local=int(tick-75*window)
            if not np.array_equal(native[window,local,79:127],sampled[k,79:127]) or not mask[window,local,79:127].all():raise ValueError("Consumed source differs from canonical plan")
            checked_frames+=1
        phases=[p["frame"] for p in original["kinematic_phases"] if event["source_interval"][0]<=p["frame"]<event["source_interval"][1]]
        mapped=np.interp(phases,np.asarray(event["time_warp"])[:,0],np.asarray(event["time_warp"])[:,1])
        if len(mapped)>1 and (np.diff(mapped)<0).any():raise ValueError("Source phase order reversed")
        phase_checks+=len(phases)
    for w in range(1,len(native)):
        if not np.array_equal(native[w-1,75:],native[w,:75]) or not np.array_equal(mask[w-1,75:],mask[w,:75]):raise ValueError("Condition overlap identity differs")
    if mask[:,:,:79].any() or mask[:,:,127:].any():raise ValueError("Root/contact or unsupported joints were promoted")
    return dict(native_reference=native,known=mask,phase=phase,normalized=normalized,scaler_known=scaler_mask,source_positions=np.array(positions)),dict(source_frames_verified=checked_frames,phase_order_checks=phase_checks,native_condition=audit,scaler_condition=soft)

def run(config,out,replay_from=None):
    out=fresh(out);data=Path(config["data_v3"]);warp=Path(config["warp"]);oldpub=read("reports/data_event_v3_summary.json");wpub=read("reports/primitive_warp_summary.json")
    for name,value in oldpub["evidence"].items():
        if sha(data/name)!=value:raise ValueError("Frozen Data V3 evidence changed")
    for name in ("main/protocol.json","main/maps.npz","main/result.json"):
        if sha(warp/name)!=wpub["evidence"][name]:raise ValueError("E1-W evidence changed")
    parent=read(data/"main/protocol.json");catalog=read(checked(parent["catalog"]));new=read(data/"main/event_catalog_v2_shadow.json")["entries"]
    assets=read(checked(parent["assets"]));scaler=np.load(checked(assets["scaler"]));scale,bias=scaler["scale"],scaler["bias"]
    maps=dict(np.load(warp/"main/maps.npz"));records=read(data/"main/music_slots.json")
    index={(r["timeline_identity"],r["slot"]["start"],r["slot"]["end"]):i for i,r in enumerate(records)}
    cfg=config["planner"];results=[];saved={};source_bindings=[]
    for case,item in enumerate(parent["timelines"]):
        timing=read(checked(item["timeline"]));payload=item["payload"]
        if any(timing[k]!=payload[k] for k in ("fps","frames","beats","slots")):raise ValueError("Actual music slots differ")
        audio_sha=payload["audio_sha256"]
        for arm,key in (("E0_legacy","T0"),("E1_legacy","T1"),("E1_W","T3")):
            banks,uids=candidates(catalog["entries"],new,timing,arm)
            for slot,options in zip(timing["slots"],banks):
                row=index[(item["semantic_identity"],slot["start"],slot["end"])]
                expected={catalog["entries"][j]["event_uid"] for j in np.flatnonzero(maps[key][row])}
                if {e["event_uid"] for e in options}!=expected:raise ValueError("Planner admission differs from frozen mathematical eligibility")
            selected=search(banks,cfg["top_k"],cfg["beam_width"])
            if selected is None:
                results.append(dict(case=case,frames=timing["frames"],arm=arm,status="NOT_EVALUABLE_EMPTY_SLOT",empty_slots=sum(not v for v in banks)));continue
            cost,sequence=selected
            consumed=_search(catalog,timing,audio_sha,top_k=cfg["top_k"],beam_width=cfg["beam_width"],fixed_events={i:e for i,e in enumerate(sequence)})[0]
            if consumed["planning_cost"]!=cost:raise ValueError("Actual Planner changed selected sequence cost")
            if arm=="E0_legacy":
                baseline=default_plan(catalog,timing,audio_sha,top_k=cfg["top_k"],beam_width=cfg["beam_width"])
                if semantic_plan_id(consumed)!=semantic_plan_id(baseline) or consumed["planning_cost"]!=baseline["planning_cost"]:raise ValueError("Research beam path differs from default")
            folder=out/(str(case)+"_"+arm);folder.mkdir();write(folder/"event_plan.json",consumed)
            write(folder/"candidates.json",dict(all_candidate_uids=uids,events=[[{k:e[k] for k in ("event_uid","source_uid","reference","source_interval","music_interval","time_warp","candidate_uid")} for e in bank] for bank in banks]))
            arrays,checks=inspect_conditions(consumed,new,scale,bias);np.savez_compressed(folder/"condition.npz",**arrays)
            # The actual default consumer recomputes identities from the unchanged reference/map.
            for event in consumed["events"]:
                if candidate_identity(event)!=event["candidate_uid"]:raise ValueError("Candidate identity changed on consumption")
            source_bindings.append(dict(case=case,arm=arm,plan=ref(folder/"event_plan.json"),candidates=ref(folder/"candidates.json"),condition=ref(folder/"condition.npz"),semantic_plan_id=semantic_plan_id(consumed)))
            results.append(dict(case=case,frames=timing["frames"],arm=arm,status="SHADOW_CONTRACT_VERIFIED",slots=len(banks),candidate_pairs=sum(map(len,banks)),cost=cost,
                plan_sha256=sha(folder/"event_plan.json"),semantic_plan_id=semantic_plan_id(consumed),**checks))
            print("SHADOW_PLAN",case,arm,len(banks),flush=True)
    result=dict(plans=results,verified_plans=sum(r["status"]=="SHADOW_CONTRACT_VERIFIED" for r in results),default_planning_changed=False,edge_forward_calls=0,native_calls=0,optimizer_calls=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=config,data_protocol=ref(data/"main/protocol.json"),warp_protocol=ref(warp/"main/protocol.json"),catalog=parent["catalog"],scaler=assets["scaler"],bindings=source_bindings,
        actual_consumer="production _search fixed-events research invocation only; default plan parity verified",learned_retrieval_qualified=False))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result:raise ValueError("Shadow planning readouts differ")
        for old in prev.glob("*_*/condition.npz"):
            same_arrays(old,out/old.parent.name/"condition.npz")
            for name in ("event_plan.json","candidates.json"):
                if read(old.parent/name)!=read(out/old.parent.name/name):raise ValueError("Plan/candidate replay differs")
        write(out/"replay.json",dict(status="FRESH_SHADOW_PLAN_CONDITION_REPLAY",optimizer_calls=0,edge_forward_calls=0,exact_arrays=True,plans_verified=result["verified_plans"]))
    print("PLANNER_INTEGRATION_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

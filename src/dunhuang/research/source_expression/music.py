"""Planner-P1: new real publisher tracks, fixed slots, no generator promotion."""
import argparse,copy,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.planning.planner import timeline,plan as default_plan,_search
from dunhuang.planning.event_plan import semantic_plan_id
from dunhuang.planning.feasible_time_map import solve
from dunhuang.research.primitive_warp.study import oracle
from dunhuang.research.planner_integration.study import candidates,search,inspect_conditions
from .common import read,commit,fresh,same_arrays

def run(config,prepared,out,replay_from=None):
    out=fresh(out);prep=read(Path(prepared)/"protocol.json")
    if prep["config"]!=config:raise ValueError("Prepared music protocol differs")
    music=read(checked(prep["music"]));data=Path(config["data_v3"]);pub=read("reports/data_event_v3_summary.json")
    for n,v in pub["evidence"].items():
        if sha(data/n)!=v:raise ValueError("Frozen Catalog source changed")
    parent=read(data/"main/protocol.json");catalog=read(checked(parent["catalog"]));new=read(data/"main/event_catalog_v2_shadow.json")["entries"]
    assets=read(checked(parent["assets"]));runtime=read(checked(assets["audio_runtime"]))
    for x in runtime["files"]+runtime.get("wheels",[])+[runtime["wheel"]]:checked(x)
    sys.path.insert(0,str(Path(runtime["import_root"]).resolve()))
    import librosa
    if librosa.__version__!="0.11.0":raise ValueError("Frozen audio detector version differs")
    scaler=np.load(checked(parent["scaler"]));cfg=config["music"];records=[];bindings=[];exhaustive_checks=0
    for case,m in enumerate(music):
        path=checked(m["audio"]);t=timeline(path,cfg["frames"]);t["audio_sha256"]=m["audio"]["sha256"];write(out/("timeline_"+str(case)+".json"),t)
        sets={}
        for arm in ("E0_legacy","E1_legacy","E1_W"):
            banks,uids=candidates(catalog["entries"],new,t,arm);sets[arm]=[{e["event_uid"] for e in bank} for bank in banks]
            if arm=="E1_W":
                for slot in t["slots"]:
                    for e in new:
                        center=e["safe_anchor"]["frame"] if e["safe_anchor"] else None
                        entry=dict(source_interval=e["source_interval"],center=center,normalization_support=e["normalization_support"])
                        s=solve(entry,slot,t["beats"]);expected=oracle(entry,slot);bounds=s["integer_domain"]
                        actual=np.arange(bounds[0],bounds[1]+1) if bounds else np.array([],int)
                        if not np.array_equal(expected,actual):raise ValueError("New-audio integer time domain differs from exhaustive oracle")
                        exhaustive_checks+=1
            selected=search(banks,cfg["top_k"],cfg["beam_width"])
            if selected is None:
                records.append(dict(case=case,arm=arm,status="NOT_EVALUABLE_EMPTY_SLOT",empty_slots=sum(not b for b in banks),candidate_pairs=sum(map(len,banks))));continue
            cost,sequence=selected
            consumed=_search(catalog,t,m["audio"]["sha256"],top_k=cfg["top_k"],beam_width=cfg["beam_width"],fixed_events={i:e for i,e in enumerate(sequence)})[0]
            if consumed["planning_cost"]!=cost:raise ValueError("Actual consumption cost differs")
            if arm=="E0_legacy":
                base=default_plan(catalog,t,m["audio"]["sha256"],top_k=cfg["top_k"],beam_width=cfg["beam_width"])
                if semantic_plan_id(base)!=semantic_plan_id(consumed) or base["planning_cost"]!=cost:raise ValueError("Default-path parity differs")
            folder=out/(str(case)+"_"+arm);folder.mkdir();write(folder/"event_plan.json",consumed);write(folder/"candidate_uids.json",uids)
            arrays,checks=inspect_conditions(consumed,new,scaler["scale"],scaler["bias"]);np.savez_compressed(folder/"condition.npz",**arrays)
            bindings.append(dict(case=case,arm=arm,plan=ref(folder/"event_plan.json"),condition=ref(folder/"condition.npz"),candidate_uids=ref(folder/"candidate_uids.json")))
            records.append(dict(case=case,arm=arm,status="SHADOW_CONTRACT_VERIFIED",frames=cfg["frames"],slots=len(banks),candidate_pairs=sum(map(len,banks)),cost=cost,
                semantic_plan_id=semantic_plan_id(consumed),plan_sha256=sha(folder/"event_plan.json"),**checks))
            print("R6_MULTI_AUDIO_PLAN",case,arm,flush=True)
        if any(not a<=b for a,b in zip(sets["E1_legacy"],sets["E1_W"])):raise ValueError("E1-W lost a legacy E1 candidate")
    result=dict(plans=records,real_audio_assets=len(music),distinct_pcm=len({m["pcm_sha256"] for m in music}),distinct_publisher_works=len({m["work_uid"] for m in music}),
        exhaustive_domain_checks=exhaustive_checks,verified_plans=sum(r["status"]=="SHADOW_CONTRACT_VERIFIED" for r in records),
        generalized_scope="multi-audio engineering contract, publisher-known identities only",encoder_exposure="KNOWN_TRAIN",composition_aliases="UNKNOWN_OUTSIDE_PUBLISHER",
        optimizer_calls=0,edge_forward_calls=0,native_calls=0,default_planning_changed=False,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=config,prepared=ref(Path(prepared)/"protocol.json"),music=prep["music"],catalog=parent["catalog"],data_protocol=ref(data/"main/protocol.json"),bindings=bindings,audio_runtime=assets["audio_runtime"]))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result:raise ValueError("Multi-audio readouts differ")
        for old in prev.glob("*_*/condition.npz"):
            same_arrays(old,out/old.parent.name/"condition.npz")
            for name in ("event_plan.json","candidate_uids.json"):
                if read(old.parent/name)!=read(out/old.parent.name/name):raise ValueError("Multi-audio plan replay differs")
        for old in prev.glob("timeline_*.json"):
            if read(old)!=read(out/old.name):raise ValueError("Real audio timeline replay differs")
        write(out/"replay.json",dict(status="FRESH_REAL_MULTI_AUDIO_REPLAY",optimizer_calls=0,edge_forward_calls=0,plans_verified=result["verified_plans"],exact_arrays=True))
    print("R6_MULTI_AUDIO_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--prepared",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.prepared,a.out,a.replay_from)

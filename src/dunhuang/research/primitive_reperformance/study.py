"""Fixed complete-event single-window comparison, then conditional whole songs."""
import argparse,copy,subprocess,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.planning.event_plan import anchors
from dunhuang.research.source_capacity.common import read,commit,fresh
from .metrics import measure,response,decide
ARRAYS=("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy")

def worker(q,out,assets,full=False):
    out.mkdir();q=dict(q,output=str(out.resolve()));write(out/"request.json",q)
    module="dunhuang.research.primitive_reperformance.worker" if full else "dunhuang.edge_backend"
    for python,mod in ((assets["edge_python"],module),(assets["native_python"],"dunhuang.physical")):
        with (out/(mod.split(".")[-1]+".log")).open("w") as log:
            subprocess.run([python,"-m",mod,str((out/"request.json").resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)

def run(cfg,prepared,out,replay_from=None):
    out=fresh(out);prep=read(Path(prepared)/"protocol.json")
    if prep["config"]!=cfg:raise ValueError("Registered single-event configuration changed")
    assets=read(checked(prep["assets"]));sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    pairs=read(checked(prep["pairs"]));music=read(checked(prep["music"]));records=[];bindings=[];states={}
    for pair in pairs:
        plans={k:read(checked(v)) for k,v in pair["plans"].items()}
        for audio in music:
            variants=[("baseline",None)]+[(arm,s) for arm in cfg["arms"] if arm!="baseline" for s in ("A","B")]
            for arm,label in variants:
                name=str(pair["pair"])+"_"+str(audio["case"])+"_"+arm+("_"+label if label else "")
                plan=copy.deepcopy(plans[label or "A"]);plan["audio_sha256"]=audio["audio"]["sha256"]
                file=out/(name+"_plan.json");write(file,plan)
                q=dict(assets,features=audio["features"],audio=audio["audio"],frames=pair["frames"],windows=1,
                    mode="custom_none" if arm=="baseline" else "temporal",seed=cfg["seed"]+pair["pair"],
                    anchors=[] if arm=="baseline" else anchors(plan),event_plan=ref(file),trajectory_radius=12,trajectory_strength=.55,overlap_policy="mean",
                    generation_stage="R8_COMPLETE_EVENT_SINGLE_WINDOW",research_reperformance_v1=arm in ("S1","S2","S3"),reperformance_config=cfg["guide"],condition_group=arm if arm in ("S1","S2","S3") else "S3")
                folder=out/name;worker(q,folder,assets,arm in ("S1","S2","S3"))
                x=np.load(folder/"native.npy");physical=read(folder/"physical.json");generation=read(folder/"generation.json")
                if len(x)!=pair["frames"] or generation["weights_before"]!=generation["weights_after"]:raise ValueError("Complete event or frozen weights differ")
                sources={k:measure(x,p,cfg["guide"]) for k,p in plans.items()}
                row=dict(pair=pair["pair"],cohort=pair["cohort"],music=audio["case"],arm=arm,source_label=label,frames=len(x),native_ok=physical["absolute"]["ok"],
                    native_reasons=physical["absolute"]["reasons"],native_audit=physical["audit"],noise_sha256=generation["sampler"]["noise_sha256"],
                    model_digest=generation["weights_before"],post_generation_pose_paste=generation["sampler"]["post_generation_pose_paste"],
                    sources=sources,source=sources.get(label),other_source=sources.get("B" if label=="A" else "A") if label else None)
                if arm in ("S1","S2","S3"):
                    audit=read(folder/"reperformance_guide.json");row["reference_control"]=dict(domain_rejections=sum(x["domain_rejected_joint_frames"] for x in audit["steps"]),
                        mean_accepted_fraction=float(np.mean([x["mean_accepted_fraction"] for x in audit["steps"]])),phase_count=audit["reference"]["phase_count"],
                        source_joint_frames=audit["reference"]["source_joint_frames"],controlled_joints=audit["reference"]["controlled_joints"])
                records.append(row);bindings.append(dict(name=name,request=ref(folder/"request.json"),native=ref(folder/"native.npy"),fk=ref(folder/"native_fk.npy"),physical=ref(folder/"physical.json"),plan=ref(file)))
                states[(pair["pair"],audio["case"],arm,label)]=x
                if replay_from:
                    previous=Path(replay_from)/name
                    for key in ARRAYS:
                        if not np.array_equal(np.load(previous/key),np.load(folder/key)):raise ValueError("Single-event fresh generation replay differs: "+name+"/"+key)
                    if read(previous/"physical.json")!=physical:raise ValueError("Single-event Native replay differs")
                print("R8_SINGLE_EVENT",name,len(x),physical["absolute"]["ok"],flush=True)
    if len(records)!=cfg["full_generation_budget"]:raise ValueError("Registered single-event budget differs")
    if len({r["model_digest"] for r in records})!=1:raise ValueError("Multiple EDGE checkpoints used")
    for pair in pairs:
        if len({r["noise_sha256"] for r in records if r["pair"]==pair["pair"]})!=1:raise ValueError("Source/music/arm noise mismatch")
    music_response=[]
    for pair in pairs:
        for source in ("A","B"):
            a,b=[states[(pair["pair"],m,"S3",source)] for m in cfg["music_cases"]]
            delta=response(a,b);music_response.append(dict(pair=pair["pair"],source_label=source,local_pose_change_rad=delta,passed=delta>=cfg["gate"]["minimum_music_response_rad"]))
    qualification=decide(records,cfg["gate"]);n=qualification["denominator"]
    passed=n==cfg["pairs"]*2*len(music) and qualification["source_passes"]==n and qualification["native_passes"]==n and all(v["passed"] for v in music_response)
    result=dict(records=records,qualification=qualification,music_response=music_response,single_event_gate_passed=passed,
        whole_song_authorized_by_single_gate=passed,neural_weight_updates=0,complete_event_generations=len(records),
        operational_qualified=False,human_reviewed=False,cultural_authenticity="UNKNOWN",new_whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,prepared=ref(Path(prepared)/"protocol.json"),bindings=bindings))
    if replay_from:
        if read(Path(replay_from)/"result.json")!=result:raise ValueError("Single-event readout replay differs")
        write(out/"replay.json",dict(status="FRESH_COMPLETE_EVENT_GENERATION_AND_NATIVE_REPLAY",full_generations=len(records),exact_arrays=len(records)*len(ARRAYS),
            neural_weight_updates=0,inference_projections_rerun=True))
    print("R8_SINGLE_GATE",passed,qualification["source_passes"],qualification["native_passes"],n,flush=True)

def whole(cfg,prepared,single,out,replay_from=None,bridge=None):
    out=fresh(out);result=read(Path(single)/"result.json")
    if not result["single_event_gate_passed"] or bridge is None or not read(Path(bridge)/"result.json")["bridge_gate_passed"]:
        write(out/"result.json",dict(status="BLOCKED_SINGLE_OR_BRIDGE_GATE",new_whole_song_generations=0,operational_qualified=False,
            source_passes=result["qualification"]["source_passes"],native_passes=result["qualification"]["native_passes"],denominator=result["qualification"]["denominator"]))
        write(out/"protocol.json",dict(code_commit=commit(),single_result=ref(Path(single)/"result.json"),config=cfg));return
    prior=Path(cfg["prior_song"]);mp=read(prior/"music/protocol.json");inventory=read(checked(mp["inventory"]));plans=read(checked(mp["generation_plans"]))
    features=read(prior/"features/protocol.json");assets=read(checked(read(Path(prepared)/"protocol.json")["assets"]))
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    side=read(checked(read(Path(prepared)/"protocol.json")["phase_catalog"]))["entries"];index={e["event_uid"]:e for e in side}
    rows=[];bindings=[]
    for case in cfg["whole_song"]["cases"]:
        item=inventory[case];base=read(checked(plans[case]["plans"]["A"]))
        for e in base["events"]:
            original=index[e["event_uid"]]
            e.update(observed_joints=original["observed_joints"],full_phases=original["kinematic_phases"],key_motion=original["key_motion"],root_translation_observed=False,contact_observed=False)
        file=out/(str(case)+"_plan.json");write(file,base)
        for arm in cfg["whole_song"]["arms"]:
            folder=out/(str(case)+"_"+arm)
            q=dict(assets,audio=item["audio"],features=features["songs"][case]["features"],frames=item["clock"]["frames"],windows=item["clock"]["windows"],
                mode="custom_none" if arm=="baseline" else "temporal",seed=cfg["seed"],anchors=[] if arm=="baseline" else anchors(base),event_plan=ref(file),
                trajectory_radius=12,trajectory_strength=.55,overlap_policy="mean",generation_stage="R8_QUALIFIED_SINGLE_EVENT_TO_WHOLE_RESEARCH",
                research_reperformance_v1=arm in ("S1","S2","S3"),reperformance_config=cfg["guide"],condition_group=arm if arm in ("S1","S2","S3") else "S3")
            worker(q,folder,assets,arm in ("S1","S2","S3"));x=np.load(folder/"native.npy");physical=read(folder/"physical.json")
            rows.append(dict(case=case,arm=arm,frames=len(x),native_ok=physical["absolute"]["ok"],native_audit=physical["audit"],source=measure(x,base,cfg["guide"])))
            bindings.append(dict(case=case,arm=arm,native=ref(folder/"native.npy"),fk=ref(folder/"native_fk.npy"),physical=ref(folder/"physical.json"),request=ref(folder/"request.json"),plan=ref(file)))
            if replay_from:
                for name in ARRAYS:
                    if not np.array_equal(np.load(folder/name),np.load(Path(replay_from)/folder.name/name)):raise ValueError("Whole-song research replay differs")
            print("R8_WHOLE_EVENT",case,arm,len(x),physical["absolute"]["ok"],flush=True)
    result=dict(status="SINGLE_GATE_PASSED_WHOLE_RESEARCH_EXECUTED",records=rows,new_whole_song_generations=len(rows),operational_qualified=False,neural_weight_updates=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),single_result=ref(Path(single)/"result.json"),config=cfg,bindings=bindings))
    if replay_from:
        if read(Path(replay_from)/"result.json")!=result:raise ValueError("Whole-song readouts differ")
        write(out/"replay.json",dict(status="FRESH_WHOLE_EVENT_REPLAY",full_generations=len(rows),exact_arrays=len(rows)*len(ARRAYS)))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--prepared",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");p.add_argument("--single");p.add_argument("--bridge");a=p.parse_args()
    if a.single:whole(read(a.config),a.prepared,a.single,a.out,replay_from=a.replay_from,bridge=a.bridge)
    else:run(read(a.config),a.prepared,a.out,a.replay_from)

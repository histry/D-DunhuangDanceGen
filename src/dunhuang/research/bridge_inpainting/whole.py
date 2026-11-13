"""Conditional full-song bridge sweep; only locally qualified edits are accepted."""
import argparse,copy,subprocess,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.research.primitive_reperformance.environment import worker_environment
from dunhuang.research.source_capacity.common import read,commit,fresh
from .audit import context_binding
from .study import audit_native,region,source_error

def run(cfg,rconfig,whole,out,replay_from=None):
    out=fresh(out);parent=read(Path(whole)/"result.json")
    if parent["new_whole_song_generations"]==0:
        result=dict(status="BLOCKED_UPSTREAM_SINGLE_OR_BRIDGE_GATE",new_local_generations=0,new_whole_song_generations=0,operational_qualified=False)
        write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),whole_result=ref(Path(whole)/"result.json")));return
    assets=read(rconfig["assets"]);sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    p=read(Path(whole)/"protocol.json");feature=read(Path(rconfig["prior_song"])/"features/protocol.json");scaler=np.load(checked(assets["scaler"]));records=[];bindings=[]
    for binding in p["bindings"]:
        if binding["arm"]!="S3":continue
        case=binding["case"];original=np.load(checked(binding["native"]));edited=original.copy();plan=read(checked(binding["plan"]));N=len(original)
        music=np.load(checked(feature["songs"][case]["features"]));attempts=[];union=np.zeros(N,bool)
        for index in range(1,len(music)):
            boundary=75*index;start=boundary-75;a,b=boundary-15,boundary+15;local=original[start:start+150]
            if len(local)!=150:raise ValueError("Registered whole Bridge lacks actual contexts")
            folder=out/(str(case)+"_"+str(boundary));folder.mkdir()
            values,mask,unsupported=context_binding(local,scaler["scale"],scaler["bias"],cfg)
            if unsupported:
                attempts.append(dict(boundary=boundary,status="REJECTED_ACTUAL_CONTEXT_OUTSIDE_NORMALIZER",unsupported_coordinates=unsupported));continue
            context=folder/"context.npz";np.savez_compressed(context,values=values,mask=mask)
            file=folder/"music.npy";np.save(file,music[index-1:index])
            q=dict(assets,output=str(folder.resolve()),audio=feature["songs"][case]["audio"],features=ref(file),frames=150,windows=1,mode="anchored",anchors=[],
                seed=cfg["seed"]+case*1000+index,overlap_policy="mean",research_bridge_v1=True,bridge_context=ref(context),bridge_source_condition=True,
                event_plan=binding["plan"],reference_offset=start,edit_interval=cfg["edit"],condition_group="S3_bridge",reperformance_config=rconfig["guide"],
                generation_stage="R8_CONDITIONAL_FULL_SONG_BRIDGE")
            write(folder/"request.json",q)
            for python,module in ((assets["edge_python"],"dunhuang.research.bridge_inpainting.worker"),(assets["native_python"],"dunhuang.physical")):
                with (folder/(module.split(".")[-1]+".log")).open("w") as log:subprocess.run([python,"-m",module,str((folder/"request.json").resolve())],stdout=log,stderr=subprocess.STDOUT,check=True,env=worker_environment())
            new=np.load(folder/"native.npy");trial=edited.copy();trial[a:b]=new[60:90]
            before_fk,before_audit=audit_native(edited);after_fk,after_audit=audit_native(trial);_,roi=audit_native(trial[start:start+150])
            before=region(before_fk,boundary-6,boundary+6);after=region(after_fk,boundary-6,boundary+6)
            joins=all(region(after_fk,t-3,t+3)["maximum"]<=region(before_fk,t-3,t+3)["maximum"]+cfg["gate"]["join_jerk_nonregression"] for t in (a,b))
            fidelity=source_error(trial,plan,a,b)<=source_error(edited,plan,a,b)+cfg["gate"]["source_nonregression_rad"]
            accepted=roi["absolute"]["ok"] and joins and fidelity and after["p95"]<=before["p95"]*(1-cfg["gate"]["boundary_p95_reduction"])
            if accepted:edited=trial;union[a:b]=True
            attempts.append(dict(boundary=boundary,status="ACCEPTED_LOCAL_REPAIR" if accepted else "REJECTED_LOCAL_REPAIR",boundary_before=before,boundary_after=after,
                joins_protected=joins,source_preserved=fidelity,roi_native_ok=roi["absolute"]["ok"],request=ref(folder/"request.json")))
            if replay_from:
                previous=Path(replay_from)/folder.name
                for name in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy"):
                    if not np.array_equal(np.load(folder/name),np.load(previous/name)):raise ValueError("Whole bridge sampling replay differs")
        if not np.array_equal(edited[~union],original[~union]):raise ValueError("Whole bridge changed protected motion")
        folder=out/("case_"+str(case));folder.mkdir();points,physical=audit_native(edited);np.save(folder/"native.npy",edited);np.save(folder/"fk.npy",points);write(folder/"physical.json",physical)
        records.append(dict(case=case,attempts=attempts,accepted=sum(a["status"]=="ACCEPTED_LOCAL_REPAIR" for a in attempts),native_ok=physical["absolute"]["ok"],source=source_error(edited,plan,0,N),protected_unchanged=True))
        bindings.append(dict(case=case,native=ref(folder/"native.npy"),fk=ref(folder/"fk.npy"),physical=ref(folder/"physical.json")))
        if replay_from:
            if not np.array_equal(edited,np.load(Path(replay_from)/folder.name/"native.npy")):raise ValueError("Whole protected assembly replay differs")
    result=dict(status="JOINT_RESEARCH_COMPLETED",records=records,new_local_generations=sum(a["status"]!="REJECTED_ACTUAL_CONTEXT_OUTSIDE_NORMALIZER" for r in records for a in r["attempts"]),
        new_whole_song_generations=0,operational_qualified=False,event_boundaries_audited_by_final_native=True,
        scope="single pass at all computational seams; music-event boundaries are distinct and not automatically edited")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),whole_protocol=ref(Path(whole)/"protocol.json"),config=cfg,bindings=bindings))
    if replay_from:
        if read(Path(replay_from)/"result.json")!=result:raise ValueError("Whole bridge readouts differ")
        write(out/"replay.json",dict(status="FRESH_JOINT_WHOLE_BRIDGE_REPLAY",new_local_generations=result["new_local_generations"]))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--reperformance-config",required=True);p.add_argument("--whole",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),read(a.reperformance_config),a.whole,a.out,a.replay_from)

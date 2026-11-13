"""Full-length M1/M2 and fixed-encoder M3 Shadow; zero generation calls."""
import argparse,sys
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.music import clock_plan
from dunhuang.planning.planner import timeline as legacy_timeline,_search,plan as default_plan
from dunhuang.planning.event_plan import semantic_plan_id
from dunhuang.planning.feasible_time_map import solve
from dunhuang.planning.time_map import source_positions,motion_features74
from dunhuang.generation.soft_reference import resample_native
from dunhuang.research.primitive_warp.study import oracle
from dunhuang.research.planner_integration.study import candidates,search,inspect_conditions
from dunhuang.research.source_capacity.common import read,commit,fresh,same_arrays
from .structure import timeline as acoustic_timeline
def runtime(assets):
    r=read(checked(assets["audio_runtime"]))
    for x in r["files"]+r.get("wheels",[])+[r["wheel"]]:checked(x)
    sys.path.insert(0,str(Path(r["import_root"]).resolve()));sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    import librosa
    if librosa.__version__!="0.11.0":raise ValueError("Frozen audio runtime differs")
def shortlist(bank):
    ordered=sorted(bank,key=lambda e:(-e["scores"]["heuristic"],e["event_uid"]));seen=set();result=[]
    for e in ordered:
        if e["source_uid"] not in seen:result.append(e);seen.add(e["source_uid"])
    return result
def model_score(model,sync,norm,music,mask,events,slot,sources):
    a,b=slot["start"],slot["end"];motions=[];masks=[]
    for e in events:
        key=e["reference"]["sha256"]
        if key not in sources:sources[key]=np.load(checked(e["reference"]))
        pos=source_positions(e,np.arange(a,b));motion,observed=motion_features74(resample_native(sources[key],pos))
        observed[0]=False # unknown derivative at an isolated mapped slot boundary
        motions.append(motion);masks.append(observed)
    music=torch.from_numpy(music[a:b]).float()[None];mm=torch.from_numpy(mask[a:b])[None]
    dm=torch.from_numpy(np.stack(masks));motion=torch.from_numpy(np.stack(motions)).float()
    def normed(x,m,k):return (x-torch.tensor(norm[k]["mean"]).float())/torch.tensor(norm[k]["std"]).float()*m
    with torch.no_grad():
        mv,mt,mvalid=model.music(normed(music,mm,"music"),mm);dv,dt,dvalid=model.motion(normed(motion,dm,"motion"),dm)
        score=(mv*dv).sum(-1);local=sync.sync_score(mt.expand(len(events),-1,-1),dt,mvalid.expand(len(events),-1)&dvalid)
    return score.numpy(),local.numpy()
def run(config,out,replay_from=None):
    import soundfile as sf,hashlib
    out=fresh(out);assets=read(config["assets"]);runtime(assets);torch.set_num_threads(4)
    from evaluation.paired_alignment_observations_v3 import full_audio_observations
    parent=read(Path(config["data_v3"])/"main/protocol.json")
    old=read(checked(parent["catalog"]));new=read(Path(config["data_v3"])/"main/event_catalog_v2_shadow.json")["entries"];scale=np.load(checked(parent["scaler"]))
    # Freeze existing SDMR encoders and sync head; never use authorization to promote them.
    cp=read(config["shadow"]["clock_protocol"])
    if cp["global_checkpoint"]["sha256"]!=config["shadow"]["global_checkpoint_sha256"] or cp["sync_checkpoint"]["sha256"]!=config["shadow"]["sync_checkpoint_sha256"]:raise ValueError("Registered frozen SDMR checkpoints differ")
    from dunhuang.alignment.model import RetrievalModel
    global_saved=torch.load(checked(cp["global_checkpoint"]),map_location="cpu",weights_only=True);sync_saved=torch.load(checked(cp["sync_checkpoint"]),map_location="cpu",weights_only=True)
    if sync_saved["global_checkpoint_sha256"]!=cp["global_checkpoint"]["sha256"] or global_saved["norm"]!=sync_saved["norm"]:raise ValueError("Frozen SDMR identities differ")
    model=RetrievalModel().eval();model.load_state_dict(global_saved["state"]);sync=RetrievalModel().eval();sync.load_state_dict(sync_saved["state"])
    for branch in ("music","motion"):
        if any(not torch.equal(v,getattr(sync,branch).state_dict()[k]) for k,v in getattr(model,branch).state_dict().items()):raise ValueError("Global/sync encoder parameters differ")
    model.requires_grad_(False);sync.requires_grad_(False)
    inventory=[];features=[];timings=[];records=[];bindings=[];shadow=[];arrays={};sources={}
    for case,name in enumerate(config["files"]):
        path=Path(config["music_root"])/name;wave,sr=sf.read(path,dtype="float32",always_2d=True);clock=clock_plan(len(wave),sr);frames=clock["frames"]
        mono=wave.mean(1);digest=hashlib.sha256(mono.tobytes()+str(sr).encode()).hexdigest()
        values,mask,receipt=full_audio_observations(path);missing=frames-len(values)
        if not 0<=missing<=1:raise ValueError("Full-audio detector clock differs by more than tail quantization")
        if missing:values=np.pad(values,((0,missing),(0,0)));mask=np.pad(mask,((0,missing),(0,0)),constant_values=False)
        values=values[:frames].astype(np.float32);mask=mask[:frames].astype(bool)
        item=dict(case=case,name=name,audio=ref(path),pcm_mono_sha256=digest,sample_rate=sr,samples=len(wave),channels=wave.shape[1],clock=clock,scope=config["scope"])
        inventory.append(item);features.append((values,mask));arrays[str(case)+"__music50"]=values;arrays[str(case)+"__music_mask"]=mask;write(out/(str(case)+"_audio_receipt.json"),receipt)
        legacy=legacy_timeline(path,frames);legacy["audio_sha256"]=item["audio"]["sha256"];acoustic=acoustic_timeline(legacy,values,config["structural_slots"])
        timings.append((legacy,acoustic))
        for timing_name,t in zip(config["planner"]["timelines"],(legacy,acoustic)):
            write(out/(str(case)+"_"+timing_name+"_timeline.json"),t);sets={}
            for arm in config["planner"]["arms"]:
                banks,uids=candidates(old["entries"],new,t,arm);sets[arm]=[{e["event_uid"] for e in b} for b in banks]
                exhaustive=0
                if arm=="E1_W":
                    for slot in t["slots"]:
                        for e in new:
                            entry=dict(source_interval=e["source_interval"],center=e["safe_anchor"]["frame"] if e["safe_anchor"] else None,normalization_support=e["normalization_support"])
                            d=solve(entry,slot,t["beats"]);bounds=d["integer_domain"];actual=np.arange(bounds[0],bounds[1]+1) if bounds else np.array([],int)
                            if not np.array_equal(oracle(entry,slot),actual):raise ValueError("Full-song time-domain certificate differs")
                            exhaustive+=1
                chosen=search(banks,config["planner"]["top_k"],config["planner"]["beam_width"])
                if chosen is None:
                    records.append(dict(case=case,timeline=timing_name,arm=arm,status="NOT_EVALUABLE_EMPTY_SLOT",slots=len(banks),empty_slots=sum(not b for b in banks),candidate_pairs=sum(map(len,banks))));continue
                cost,sequence=chosen
                plan=_search(old,t,item["audio"]["sha256"],top_k=5,beam_width=3,fixed_events={i:e for i,e in enumerate(sequence)})[0]
                if plan["planning_cost"]!=cost or plan["frames"]!=frames:raise ValueError("Whole-plan consumer differs")
                if arm=="E0_legacy":
                    baseline=default_plan(old,t,item["audio"]["sha256"],top_k=5,beam_width=3)
                    if semantic_plan_id(baseline)!=semantic_plan_id(plan) or baseline["planning_cost"]!=cost:raise ValueError("Full-song frozen baseline parity differs")
                folder=out/(str(case)+"_"+timing_name+"_"+arm);folder.mkdir();write(folder/"event_plan.json",plan);write(folder/"candidate_uids.json",uids)
                condition,checks=inspect_conditions(plan,new,scale["scale"],scale["bias"]);np.savez_compressed(folder/"condition.npz",**condition)
                bindings.append(dict(case=case,timeline=timing_name,arm=arm,plan=ref(folder/"event_plan.json"),condition=ref(folder/"condition.npz")))
                records.append(dict(case=case,timeline=timing_name,arm=arm,status="FULL_SONG_SHADOW_CONTRACT",frames=frames,slots=len(banks),candidate_pairs=sum(map(len,banks)),
                    unique_events=len({e["event_uid"] for e in sequence}),unique_sources=len({e["reference"]["sha256"] for e in sequence}),repeated_events=len(sequence)-len({e["event_uid"] for e in sequence}),
                    cost=cost,semantic_plan_id=semantic_plan_id(plan),exhaustive_domains=exhaustive,**checks))
                print("R7_FULL_PLAN",case,timing_name,arm,frames,flush=True)
            if any(not a<=b for a,b in zip(sets["E1_legacy"],sets["E1_W"])):raise ValueError("E1-W lost E1 candidates")
    # Same candidate set and same absolute interval for music-swap controls; no semantic labels.
    for case,(legacy,t) in enumerate(timings):
        banks,_=candidates(old["entries"],new,t,"E1_W")
        for si,(slot,bank) in enumerate(zip(t["slots"],banks)):
            events=shortlist(bank)
            if not events:shadow.append(dict(case=case,slot=si,status="NULL_NO_LEGAL_EVENT"));continue
            score,local=model_score(model,sync,global_saved["norm"],*features[case],events,slot,sources)
            swapped=(case+1)%3
            if slot["end"]<=len(features[swapped][0]):
                other,_=model_score(model,sync,global_saved["norm"],*features[swapped],events,slot,sources)
            else:other=score.copy() # explicit tail scope flag below
            arrays[str(case)+"__"+str(si)+"__global"]=score;arrays[str(case)+"__"+str(si)+"__sync"]=local;arrays[str(case)+"__"+str(si)+"__music_swap"]=other
            order=np.argsort(-score,kind="stable");energy=np.argsort([-e["scores"]["heuristic"] for e in events],kind="stable")
            shadow.append(dict(case=case,slot=si,status="SHADOW_UNQUALIFIED",candidates=len(events),top_event=events[int(order[0])]["event_uid"],energy_top_event=events[int(energy[0])]["event_uid"],
                ranking_differs_from_energy=bool(order[0]!=energy[0]),music_swap_changed_top=bool(score.argmax()!=other.argmax()),music_swap_evaluable=slot["end"]<=len(features[swapped][0]),
                semantic_rejection="NOT_CALIBRATED",candidate_scope="one best legal event per source; not full catalog retrieval"))
        print("R7_SHADOW_MUSIC",case,flush=True)
    # Register two legal source plans for each full song; no model-dependent selection.
    generation_plans=[]
    for case,item in enumerate(inventory):
        t=timings[case][1];banks,_=candidates(old["entries"],new,t,"E1_W")
        if any(not b for b in banks):
            generation_plans.append(dict(case=case,status="BLOCKED_EMPTY_E1W_SLOT"));continue
        first=search(banks,5,3)[1];alternative=[]
        for bank,e in zip(banks,first):
            possible=[v for v in shortlist(bank) if v["reference"]["sha256"]!=e["reference"]["sha256"]]
            if not possible:break
            alternative.append(possible[0])
        if len(alternative)!=len(first):generation_plans.append(dict(case=case,status="BLOCKED_NO_SOURCE_CONTROL"));continue
        paths={}
        for label,seq in (("A",first),("B",alternative)):
            plan=_search(old,t,item["audio"]["sha256"],top_k=5,beam_width=3,fixed_events={i:e for i,e in enumerate(seq)})[0]
            file=out/(str(case)+"_generation_"+label+".json");write(file,plan);paths[label]=ref(file)
        generation_plans.append(dict(case=case,status="FULL_EVENT_PLANS_BOUND",plans=paths))
    np.savez_compressed(out/"arrays.npz",**arrays);write(out/"inventory.json",inventory);write(out/"shadow_private.json",shadow);write(out/"generation_plans.json",generation_plans)
    public_shadow=[{k:v for k,v in row.items() if k not in ("top_event","energy_top_event")} for row in shadow]
    result=dict(scope=config["scope"],full_song_frames=[r["clock"]["frames"] for r in inventory],distinct_pcm=len({r["pcm_mono_sha256"] for r in inventory}),
        plans=records,verified_plans=len(bindings),shadow=public_shadow,music_swap_changed_top=sum(r.get("music_swap_changed_top",False) for r in shadow),
        shadow_ranking_differs_from_energy=sum(r.get("ranking_differs_from_energy",False) for r in shadow),default_retrieval_changed=False,model_fit_calls=0,human_reviewed=False,operational_qualified=False,
        phrase_ground_truth=False,generation_cases_bound=sum(r["status"]=="FULL_EVENT_PLANS_BOUND" for r in generation_plans))
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=config,assets=ref(config["assets"]),inventory=ref(out/"inventory.json"),bindings=bindings,
        generation_plans=ref(out/"generation_plans.json"),global_checkpoint=cp["global_checkpoint"],sync_checkpoint=cp["sync_checkpoint"],catalog=parent["catalog"],scaler=parent["scaler"]))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result or read(prev/"shadow_private.json")!=shadow:raise ValueError("Full music readout replay differs")
        same_arrays(prev/"arrays.npz",out/"arrays.npz")
        for file in prev.glob("*_*/condition.npz"):same_arrays(file,out/file.parent.name/"condition.npz")
        for file in prev.glob("*_*/event_plan.json"):
            if read(file)!=read(out/file.parent.name/file.name):raise ValueError("Whole-song plans differ")
        write(out/"replay.json",dict(status="FRESH_FULL_SONG_M1_M2_M3_REPLAY",optimizer_calls=0,plans=len(bindings),exact_arrays=True))
    print("R7_FULL_MUSIC_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

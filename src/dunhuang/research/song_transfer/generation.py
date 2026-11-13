"""Existing matched sampler and Temporal V0.5 only, on complete songs."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import matrices
from dunhuang.planning.event_plan import anchors
from dunhuang.generation.soft_reference import build_native_event_reference
from dunhuang.research.source_capacity.common import read,commit,fresh
def following(native,plan):
    value,mask,phase,_=build_native_event_reference(plan,len(native));ticks=np.arange(len(native));w=np.minimum(ticks//75,len(value)-1);local=ticks-75*w
    reference=value[w,local];known=mask[w,local,79:127].reshape(len(native),8,6).all(-1)
    a=matrices(native[:,79:127].reshape(-1,8,6));b=matrices(reference[:,79:127].reshape(-1,8,6))
    error=Rotation.from_matrix((a.swapaxes(-1,-2)@b).reshape(-1,3,3)).magnitude().reshape(-1,8)
    return float(error[known].mean())
def worker(request,out,assets):
    out.mkdir();request=dict(request,output=str(out.resolve()));write(out/"request.json",request)
    for python,module in ((assets["edge_python"],"dunhuang.edge_backend"),(assets["native_python"],"dunhuang.physical")):
        with (out/(module.split(".")[-1]+".log")).open("w") as log:
            subprocess.run([python,"-m",module,str((out/"request.json").resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
def run(config,music,features,out,replay_from=None):
    out=fresh(out);assets=read(config["assets"]);mp=read(Path(music)/"protocol.json");inventory=read(checked(mp["inventory"]));plans=read(checked(mp["generation_plans"]))
    features_file=Path(features)/"protocol.json";features=read(features_file);records=[];bindings=[];states={}
    for case,item in enumerate(inventory):
        pair=plans[case]
        if pair["status"]!="FULL_EVENT_PLANS_BOUND":raise ValueError("Full-song source controls not constructible")
        plan_a=read(checked(pair["plans"]["A"]));plan_b=read(checked(pair["plans"]["B"]))
        paths=features["songs"][case]
        if paths["audio"]["sha256"]!=item["audio"]["sha256"] or paths["clock"]!=item["clock"]:raise ValueError("Full Jukebox audio binding differs")
        values=np.load(checked(paths["features"]));clock=item["clock"]
        if values.shape!=(clock["windows"],150,4800):raise ValueError("Feature windows are not whole song")
        arms=list(config["generation"]["arms"])
        if case==config["generation"]["extra_permuted_baseline_case"]:arms.append("music_order_control")
        for arm in arms:
            file=checked(paths["features"])
            if arm=="music_order_control":
                order=np.random.default_rng(config["generation"]["seed"]).permutation(len(values));file=out/"permuted_music.npy";np.save(file,values[order]);write(out/"permuted_music_order.json",order.tolist())
            controlled=arm in ("temporal_A","temporal_B");plan=plan_a if arm=="temporal_A" else plan_b
            request=dict(assets,features=ref(file),audio=item["audio"],frames=clock["frames"],windows=clock["windows"],seed=config["generation"]["seed"],
                mode="temporal" if controlled else "custom_none",anchors=anchors(plan) if controlled else [],trajectory_radius=config["generation"]["trajectory_radius"],
                trajectory_strength=config["generation"]["trajectory_strength"],overlap_policy="mean",beat_frames=read(Path(music)/(str(case)+"_acoustic_v1_timeline.json"))["beats"],
                generation_stage="EXPOSED_FULL_SONG_EXISTING_TEMPORAL_V05",research_condition_policy="no_learned_adapter",music_intervention="permuted full feature-window order" if arm=="music_order_control" else None)
            folder=out/(str(case)+"_"+arm);worker(request,folder,assets)
            native=np.load(folder/"native.npy");physics=read(folder/"physical.json");following_a=following(native,plan_a);following_b=following(native,plan_b)
            if len(native)!=clock["frames"]:raise ValueError("Output is not the complete song")
            record=dict(case=case,arm=arm,frames=len(native),windows=clock["windows"],native_ok=physics["absolute"]["ok"],fk_max_abs_m=physics["fk_max_abs_m"],
                source_A_error_rad=following_a,source_B_error_rad=following_b,source_preference_rad=following_b-following_a,
                native_audit=physics["audit"],evaluation=read(folder/"evaluation.json"),sampler=read(folder/"generation.json")["sampler"])
            records.append(record);bindings.append(dict(case=case,arm=arm,request=ref(folder/"request.json"),native=ref(folder/"native.npy"),fk=ref(folder/"native_fk.npy"),physical=ref(folder/"physical.json")))
            states[(case,arm)]=native
            if replay_from:
                previous=Path(replay_from)/folder.name
                for name in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy"):
                    if not np.array_equal(np.load(previous/name),np.load(folder/name)):raise ValueError("Full-song independent generation replay differs")
                if read(previous/"physical.json")!=physics:raise ValueError("Full-song Native replay differs")
            print("R7_FULL_GENERATION",case,arm,len(native),physics["absolute"]["ok"],flush=True)
    music_response=[]
    for a in range(3):
        for b in range(a):
            x,y=states[(a,"baseline")],states[(b,"baseline")];n=min(len(x),len(y));ra=matrices(x[:n,7:].reshape(n,24,6));rb=matrices(y[:n,7:].reshape(n,24,6))
            music_response.append(dict(case_a=a,case_b=b,common_frames=n,mean_rotation_difference_rad=float(Rotation.from_matrix((ra.swapaxes(-1,-2)@rb).reshape(-1,3,3)).magnitude().mean()),
                root_rms_difference_m=float(np.sqrt(np.square(x[:n,4:7]-y[:n,4:7]).mean())),same_seed_and_window_batch=inventory[a]["clock"]["windows"]==inventory[b]["clock"]["windows"],
                not_semantic_coordination_proof=True))
    if len(records)!=config["generation"]["budget"]:raise ValueError("Whole-song generation budget differs")
    result=dict(records=records,music_response=music_response,complete_song_generations=len(records),native_passes=sum(r["native_ok"] for r in records),model_updates=0,
        learned_adapter_used=False,R7_guidance_used=False,bilateral_generation_claimed=False,operational_qualified=False,human_reviewed=False,scope=config["scope"])
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=config,music_protocol=ref(Path(music)/"protocol.json"),features_protocol=ref(features_file),
        bindings=bindings,comparison="existing matched constrained_ddim none vs Temporal V0.5; same music and sampler noise; no primitive pasting"))
    if replay_from:
        if read(Path(replay_from)/"result.json")!=result:raise ValueError("Full-song readouts differ")
        write(out/"replay.json",dict(status="FRESH_COMPLETE_SONG_GENERATION_AND_NATIVE_REPLAY",optimizer_calls=0,full_generations_rerun=len(records),motion_arrays_exact=len(records)*6))
    print("R7_GENERATION_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--music",required=True);p.add_argument("--features",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.music,a.features,a.out,a.replay_from)

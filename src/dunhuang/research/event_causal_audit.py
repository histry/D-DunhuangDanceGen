"""G1 matched source/visibility intervention, real frozen EDGE, no training."""
import argparse,copy,json,subprocess,hashlib,os
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write,sha
from dunhuang.contracts import MotionAdapter
from dunhuang.generation.conditions import source_condition
from dunhuang.generation.event_geometry import event_geometry
from dunhuang.generation.event_visibility import intervene
from dunhuang.generation.source_pairs import validate_pair
from dunhuang.planning.primitive_contract import build as contract,verify as verify_contract
from dunhuang.generation.train import digest
from .mechanism import Runtime,assembled
from .event_metrics import event_margin,dynamics,rotation_error
from dunhuang.sampling import constrained_ddim

CONTROLS=("legacy_a","legacy_b","legacy_removed","legacy_dense_a","legacy_mask_only","legacy_corrupted",
          "observed_a","observed_b","observed_removed","observed_corrupted","zero_adapter")
GEN_CONTROLS=("legacy_a","legacy_b","legacy_removed","observed_a","observed_b","observed_removed","zero_adapter")

def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class CausalRuntime(Runtime):
    def __init__(self,assets,adapter):
        super().__init__(assets,adapter)
        if not self.native_geometry:raise ValueError("G1 requires frozen native upper8 event adapter")
        self.adapter.requires_grad_(False)
    def inject(self,module,args,music):
        if self.state.get("enabled",True):
            capture=self.state.get("capture",False)
            output=self.adapter(self.state["geometry"],music,visibility_policy=self.state["visibility"],return_trace=capture)
            if capture:
                delta,_,trace=output
                self.layers.update({k:v.detach().cpu().numpy()[0] for k,v in trace.items()})
            else:delta=output
            result=music+delta
        else:result=music
        if self.state.get("capture",False):
            self.layers["edge_music"]=music.detach().cpu().numpy()[0];self.layers["edge_condition"]=result.detach().cpu().numpy()[0]
        return result

def geometry_for(rt,row,policy):
    native=torch.from_numpy(np.load(checked(row["motion"])))[None].cuda()
    observed=torch.from_numpy(np.load(checked(row["observed"])))[None].cuda()
    phase=torch.arange(150,device="cuda").float()[None,:,None]/150
    c=source_condition(native,observed,phase,policy)
    return event_geometry(c.value,c.known,c.phase),native[0].cpu().numpy()

def choose_state(rt,name,a,b,dense):
    base=b if name.endswith("_b") else dense if name=="legacy_dense_a" else a
    control="removed" if name.endswith("removed") else "mask_only" if name=="legacy_mask_only" else "unknown_corrupted" if name.endswith("corrupted") else "original"
    rt.state.update(geometry=intervene(base,control,119),visibility="observed" if name.startswith("observed") else "legacy",
        enabled=name!="zero_adapter")

def layer_delta(a,b):
    diff=a-b
    return dict(rms=float(np.sqrt(np.mean(diff**2))),max_abs=float(np.max(np.abs(diff))),
        relative_rms=float(np.sqrt(np.mean(diff**2))/max(np.sqrt(np.mean(b**2)),1e-12)))

def sensitivity(rt,noisy,music,t,geometry,config):
    records=[];rt.state.update(visibility="observed",enabled=True,capture=False)
    generator=torch.Generator(device="cuda");generator.manual_seed(config["seed"]+991)
    g=geometry.detach().clone().requires_grad_(True);rt.state["geometry"]=g
    prediction=rt.model(noisy,music,t,cond_drop_prob=0.)
    known=(g[...,144:152]>0).repeat_interleave(6,-1).repeat(1,1,3)
    direction=torch.randn(g.shape,generator=generator,device="cuda");direction[...,144:]=0
    direction[...,:144]*=known;direction/=direction.norm().clamp_min(1e-12)
    for index in range(config["vjp_probes"]):
        v=torch.randint(0,2,prediction.shape,generator=generator,device="cuda").to(prediction.dtype)*2-1
        v/=np.sqrt(v.numel())
        grad=torch.autograd.grad((prediction*v).sum(),g,retain_graph=index+1<config["vjp_probes"])[0]
        directional=float((grad*direction).sum().detach())
        r=dict(probe=index,normalized_output_vjp_norm=float(grad.norm().detach()),
            source_value_norm=float(grad[...,:144].norm().detach()),mask_feature_norm=float(grad[...,144:152].norm().detach()),
            phase_norm=float(grad[...,152:].norm().detach()),unobserved_value_norm=float(grad[...,:144][~known].norm().detach()),finite_differences=[])
        if index==0:
            for eps in config["fd_epsilons"]:
                with torch.no_grad():
                    rt.state["geometry"]=g.detach()+eps*direction;plus=rt.model(noisy,music,t,cond_drop_prob=0.)
                    rt.state["geometry"]=g.detach()-eps*direction;minus=rt.model(noisy,music,t,cond_drop_prob=0.)
                    fd=float(((plus-minus)*v).sum()/(2*eps))
                r["finite_differences"].append(dict(epsilon=eps,autograd=directional,finite_difference=fd,absolute_error=abs(fd-directional),
                    scope="feature-space derivative, not a feasible SO3 repair witness"))
        records.append(r)
    return records

def plan_for(row,source,folder,assets,beat_frames):
    import soundfile as sf
    folder.mkdir(parents=True,exist_ok=False)
    audio,sr=sf.read(checked(row["audio"]),always_2d=True);start=round((row["start_frame"]/30+row["audio_offset_seconds"])*sr)
    crop=audio[start:start+5*sr]
    if len(crop)!=5*sr:raise ValueError("Actual five-second music support unavailable")
    sf.write(folder/"audio.wav",crop,sr,subtype="FLOAT")
    event=dict(event_uid="g1_source",source_uid=source["recording_uid"],reference=source["motion"],
        source_interval=[0,150],music_interval=[0,150],time_warp=[[0,0],[150,150]],
        primitive=dict(source_frame=75,music_frame=75,joints=list(range(12,20)),radius=12),feasibility=dict(native="UNKNOWN"))
    plan=dict(schema="event_plan_v1",fps=30,frames=150,audio_sha256=sha(folder/"audio.wav"),time_scale_limits=[.75,1.33],events=[event])
    # Audio-only detector beat timestamps; no learned phrase segmentation.
    c=contract(plan,{source["recording_uid"]:source["observed"]},beat_frames=beat_frames);verify_contract(c,plan)
    write(folder/"plan.json",plan);write(folder/"condition_contract.json",c)
    return plan,c,ref(folder/"audio.wav")

def run(assets_file,root_protocol,output,config,replay_from=None):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);assets=read(assets_file)
    source=read(checked(assets["paired_generation_manifest"]));index={r["window_uid"]:r for r in source["rows"] if r["role"] in ("train","val")}
    root=read(root_protocol);adapter_ref=root["initial_adapter"];checked(adapter_ref)
    training=read(Path(adapter_ref["path"]).parent/"protocol.json");registered=training["validation_pairs"]
    for pair in registered:validate_pair(pair,index)
    chosen=[];music_seen=set()
    for pair in registered:
        row=index[pair["target_uid"]]
        if row["music_uid"] not in music_seen:chosen.append(pair);music_seen.add(row["music_uid"])
        if len(chosen)==config["music_cases"]:break
    if len(chosen)!=config["music_cases"]:raise ValueError("Registered independent music cases unavailable")
    alternatives=[]
    for pair in chosen:
        pool=[p for p in registered if index[p["source_uid"]]["choreography_uid"]!=index[pair["source_uid"]]["choreography_uid"]]
        if not pool:raise ValueError("Source contrast cannot be fabricated")
        alternatives.append(pool[0])
    protocol=dict(schema="g1_event_condition_causal_audit_v1",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config=config,assets=ref(assets_file),adapter=adapter_ref,training_protocol=ref(Path(adapter_ref["path"]).parent/"protocol.json"),
        selected_pairs=chosen,alternative_pairs=alternatives,optimizer_calls=0,visibility_scope="legacy-trained weights; observed-mask inference diagnosis only",
        new_blind_source=False,operational_qualified=False)
    write(out/"protocol.json",protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    previous_fastpath=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
    rt=CausalRuntime(assets,adapter_ref);records=[];snapshots={};layer_arrays={};generated=[];preferences=[]
    try:
        for case,(pair,wrong) in enumerate(zip(chosen,alternatives)):
            row=index[pair["target_uid"]];sa=index[pair["source_uid"]];sb=index[wrong["source_uid"]]
            target,music,_=rt.tensors(row);a,ref_a=geometry_for(rt,sa,"completion_upper8");b,ref_b=geometry_for(rt,sb,"completion_upper8");dense,_=geometry_for(rt,sa,"dense_upper8")
            sidecar=Path(config["event_source"])/"data/sidecars"/(row["audio"]["sha256"]+".json")
            event_source=Path(config["event_source"])
            if sha(sidecar)!=read(event_source/"preflight.json")["files"]["data/sidecars/"+sidecar.name]:raise ValueError("Beat sidecar seal differs")
            events=read(sidecar)["events"]
            beat_frames=[int(e["frame"]-row["start_frame"]-round(30*row["audio_offset_seconds"])) for e in events if e["event_type"]=="beat"]
            folder=out/("case_"+str(case));_,contract_a,audio_ref=plan_for(row,sa,folder/"reference_a",assets,beat_frames)
            plan_for(row,sb,folder/"reference_b",assets,beat_frames)
            for time in config["timesteps"]:
                torch.manual_seed(config["seed"]+case*1000+time);torch.cuda.manual_seed_all(config["seed"]+case*1000+time)
                t=torch.tensor([time],device="cuda");noisy=rt.diffusion.q_sample(target,t,noise=torch.randn_like(target))
                xs=tensor_hash(noisy);ms=tensor_hash(music);base_layers=None;preds={};metrics={}
                for control in CONTROLS:
                    choose_state(rt,control,a,b,dense);rt.state["capture"]=True;rt.layers={}
                    with torch.no_grad():pred=rt.model(noisy,music,t,cond_drop_prob=0.)
                    if xs!=tensor_hash(noisy) or ms!=tensor_hash(music):raise ValueError("Causal intervention changed noise/music")
                    native=MotionAdapter.to_native(rt.normalizer.unnormalize(pred.detach().cpu())[0].numpy())
                    preds[control]=native;snapshots[str(case)+"_"+str(time)+"_"+control]=pred.detach().cpu().numpy()
                    if base_layers is None:base_layers={k:v.copy() for k,v in rt.layers.items()}
                    deltas={k:layer_delta(v,base_layers[k]) for k,v in rt.layers.items() if k in base_layers}
                    for k,v in rt.layers.items():
                        if k in base_layers:
                            layer_arrays[str(case)+"_"+str(time)+"_"+control+"_"+k+"_frame_rms"]=np.sqrt(np.mean((v-base_layers[k])**2,axis=-1))
                            if time==500:layer_arrays[str(case)+"_"+control+"_"+k+"_delta"]=v-base_layers[k]
                    metrics[control]=dict(mse=float((pred-target).square().mean()),layers=deltas,
                        normalized_velocity_rms=float((pred[:,1:]-pred[:,:-1]).square().mean().sqrt()))
                for policy in ("legacy","observed"):
                    metrics[policy+"_preference"]=event_margin(preds[policy+"_a"],preds[policy+"_b"],ref_a,ref_b)
                records.append(dict(case=case,timestep=time,metrics=metrics,noise_sha256=xs,music_sha256=ms))
                if time==500:
                    records[-1]["sensitivity"]=sensitivity(rt,noisy,music,t,a,config)
                print("G1_DENOISING",case,time,flush=True)
            for control in GEN_CONTROLS:
                gen=folder/control;gen.mkdir();choose_state(rt,control,a,b,dense);rt.state["capture"]=False
                torch.manual_seed(config["seed"]);torch.cuda.manual_seed_all(config["seed"])
                with torch.no_grad():
                    sample,sampler=constrained_ddim(rt.diffusion,music,torch.zeros_like(target),torch.zeros_like(target,dtype=torch.bool),guidance="none")
                    assembled(rt,sample,music,gen)
                request=dict(assets,frames=150,output=str(gen),anchors=[],beat_frames=[],audio=audio_ref)
                write(gen/"physical_request.json",request)
                subprocess.run([assets["native_python"],"-m","dunhuang.physical",str(gen/"physical_request.json")],check=True,stdout=subprocess.DEVNULL,env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2])+os.pathsep+os.environ.get("PYTHONPATH","")))
                native=np.load(gen/"native.npy");fk=np.load(gen/"native_fk.npy");physical=read(gen/"physical.json")
                audit=physical["audit"];continuous={k:float(v) for k,v in audit.items() if isinstance(v,(int,float))}
                generated.append(dict(case=case,control=control,seed=config["seed"],sampler=sampler,
                    native_ok=physical["absolute"]["ok"],native_failed=physical["absolute"]["reasons"],
                    continuous=continuous,dynamics=dynamics(native,fk)))
                print("G1_GENERATED",case,control,flush=True)
            for policy in ("legacy","observed"):
                ya=np.load(folder/(policy+"_a")/"native.npy");yb=np.load(folder/(policy+"_b")/"native.npy")
                zero=np.load(folder/"zero_adapter/native.npy")
                pref=event_margin(ya,yb,ref_a,ref_b);removed=event_margin(zero,zero,ref_a,ref_b)
                own=(rotation_error(ya,ref_a)+rotation_error(yb,ref_b))/2
                zero_distance=(rotation_error(zero,ref_a)+rotation_error(zero,ref_b))/2
                z_activity=next(r["dynamics"]["upper8_angular_speed_mean_radps"] for r in generated if r["case"]==case and r["control"]=="zero_adapter")
                activities=[r["dynamics"]["upper8_angular_speed_mean_radps"] for r in generated if r["case"]==case and r["control"] in (policy+"_a",policy+"_b")]
                preferences.append(dict(case=case,policy=policy,**pref,zero_source_preference=removed["preference_margin_radians"],
                    reference_gain_vs_zero_radians=zero_distance-own,activity_retention=min(activities)/max(z_activity,1e-12)))
            hashes={r["sampler"]["noise_sha256"] for r in generated if r["case"]==case}
            if len(hashes)!=1:raise ValueError("Free-running counterfactual noise differs")
        if any(p.grad is not None for p in rt.adapter.parameters()):raise ValueError("Frozen adapter acquired parameter gradients")
    finally:
        rt.close();torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    np.savez_compressed(out/"denoising.npz",**snapshots);np.savez_compressed(out/"layer_deltas.npz",**layer_arrays)
    effective=all(p["preference_margin_radians"]>config["minimum_preference_radians"] and p["reference_gain_vs_zero_radians"]>0 and p["activity_retention"]>=config["minimum_activity_retention"] for p in preferences if p["policy"]=="observed")
    result=dict(status="COMPLETED_G1_CAUSAL_AUDIT_SHADOW",denoising=records,generations=generated,source_preferences=preferences,
        generations_executed=len(generated),native_passes=sum(r["native_ok"] for r in generated),condition_effective_on_registered_cases=effective,
        boundary_generation_started=False,protected_sqp_started=False,weights_unchanged=True,optimizer_calls=0,
        trained_visibility_policy="legacy",new_visibility_policy_trained=False,semantic_source_ground_truth="UNKNOWN",
        new_blind_source=False,operational_qualified=False)
    write(out/"result.json",result)
    if replay_from:
        original=Path(replay_from)
        for name in ("denoising","layer_deltas"):
            x=np.load(original/(name+".npz"));y=np.load(out/(name+".npz"))
            if set(x.files)!=set(y.files) or any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError("G1 fresh scientific tensor replay differs")
        if read(original/"result.json")!=result:raise ValueError("G1 causal metrics replay differs")
        for case in range(config["music_cases"]):
            for control in GEN_CONTROLS:
                for name in ("normalized.npy","native.npy","native_fk.npy"):
                    if not np.array_equal(np.load(original/("case_"+str(case))/control/name),np.load(out/("case_"+str(case))/control/name)):raise ValueError("G1 real generation replay differs")
        write(out/"replay.json",dict(status="FRESH_G1_REPLAY_VERIFIED",optimizer_calls=0,real_generations_recomputed=len(generated),weights_unchanged=True))
    print("G1_COMPLETED",flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ("assets","root-protocol","out","config"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(a.assets,a.root_protocol,a.out,read(a.config),a.replay_from)
if __name__=="__main__":main()

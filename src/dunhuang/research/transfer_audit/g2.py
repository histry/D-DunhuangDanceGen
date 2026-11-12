"""Frozen G2 attribution: source -> residual -> decoder -> CFG/clip -> motion."""
import argparse,copy,json,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.alignment.clock_data import metadata
from dunhuang.contracts import MotionAdapter
from dunhuang.generation.targets import digest
from dunhuang.generation.source_following import following_loss,hard_keyframe_inputs
from dunhuang.research.source_identifiability.g2 import Runtime,physical,source_error
from dunhuang.research.event_causal_audit import layer_delta,tensor_hash
from dunhuang.sampling import constrained_ddim

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

class AuditRuntime(Runtime):
    def inject(self,module,args,music):
        result=super().inject(module,args,music)
        if self.state.get("capture"):
            self.trace["base_condition"]=music.detach().cpu().numpy()[0]
            self.trace["injected_condition"]=result.detach().cpu().numpy()[0]
        return result

def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))

def exchange(rt,uid,other,state,x,music,t,label,tensors):
    rt.adapter.load_state_dict(state);rt.adapter.eval();rt.adapter.requires_grad_(False)
    y=[];traces=[];u=[]
    for control,donor in (("A",uid),("B",other),("NULL",uid)):
        rt.batch_self([uid],[donor])
        if control=="NULL":rt.state["geometry"]=torch.zeros_like(rt.state["geometry"])
        rt.state["capture"]=True;rt.trace={}
        with torch.no_grad():pred=rt.model(x,music,t,cond_drop_prob=0.)
        y.append(pred.detach());traces.append({k:v.copy() for k,v in rt.trace.items()})
        tensors[label+"_"+control]=pred.cpu().numpy()
        rt.state["capture"]=False
        if control!="NULL":
            with torch.no_grad():u.append(rt.model(x,music,t,cond_drop_prob=1.).detach())
    if not torch.equal(u[0],u[1]):raise AssertionError("Unconditional branch unexpectedly consumes source")
    delta={k:layer_delta(traces[1][k],traces[0][k]) for k in traces[0]}
    for k in traces[0]:tensors[label+"_delta_"+k]=traces[1][k]-traces[0][k]
    cfg=[u[0]+2*(p-u[0]) for p in y];clipped=[p.clamp(-1,1) for p in cfg]
    raw_delta=(y[1]-y[0]).cpu().numpy();guided_delta=(cfg[1]-cfg[0]).cpu().numpy();clip_delta=(clipped[1]-clipped[0]).cpu().numpy()
    refs=[rt.self_data(v)[2] for v in (uid,other)]
    native=[MotionAdapter.to_native(rt.normalizer.unnormalize(p.cpu().clone())[0].numpy()) for p in clipped]
    known=rt.source(uid)[0].known[0,:,79:127].reshape(150,8,6).all(-1).numpy();hidden=~known
    record=dict(label=label,timestep=int(t.item()),input_sha256=tensor_hash(x),music_sha256=tensor_hash(music),
        layers=delta,conditional_exchange_rms=rms(raw_delta),cfg_exchange_rms=rms(guided_delta),
        clipped_exchange_rms=rms(clip_delta),clip_retention_ratio=rms(clip_delta)/max(rms(guided_delta),1e-20),
        upper8_cfg_saturation_fraction=float(torch.stack(cfg).abs()[...,79:127].gt(1).float().mean()),
        residual_to_music_rms=rms(traces[0]["residual_512"])/max(rms(traces[0]["base_condition"]),1e-20),
        output_to_residual_exchange_ratio=rms(raw_delta)/max(delta["residual_512"]["rms"],1e-20),
        unconditional_source_invariant=True,hidden_A_error=source_error(native[0],refs[0],hidden),
        hidden_A_error_with_B=source_error(native[1],refs[0],hidden),hidden_A_error_with_null=source_error(native[2],refs[0],hidden),
        hidden_B_error=source_error(native[1],refs[1],hidden),hidden_B_error_with_A=source_error(native[0],refs[1],hidden),
        observed_frames=int(known.any(-1).sum()),hidden_frames=int(hidden.any(-1).sum()))
    return record

def gradients(rt,uid,x,music,t):
    rt.batch_self([uid]);rt.state["capture"]=False;rt.adapter.requires_grad_(True)
    target,music,_,reference,observed,keys=rt.batch_self([uid])
    pred=rt.model(x,music,t,cond_drop_prob=0.)
    denoise=(pred-target).square().mean()
    follow=following_loss(pred,reference,observed,keys,torch.as_tensor(rt.scale,device="cuda"),torch.as_tensor(rt.bias,device="cuda"),"same_recording_pose_exact")
    named=list(rt.adapter.named_parameters());params=[p for n,p in named]
    ga=torch.autograd.grad(denoise,params,retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(follow,params,allow_unused=True)
    groups={}
    for name in ("input","encoder","music","attention","norm","reconstruct","output"):
        aa=[];bb=[]
        for (n,p),a,b in zip(named,ga,gb):
            if n.split(".")[0]==name:
                aa.append(torch.zeros_like(p).flatten() if a is None else a.flatten());bb.append(torch.zeros_like(p).flatten() if b is None else b.flatten())
        a=torch.cat(aa).double();b=torch.cat(bb).double();na=a.norm();nb=b.norm()
        groups[name]=dict(denoise_norm=float(na),source_norm=float(nb),cosine=float((a*b).sum()/(na*nb)) if na>0 and nb>0 else None)
    rt.adapter.requires_grad_(False);rt.adapter.zero_grad(set_to_none=True)
    return dict(timestep=int(t.item()),denoise_mse=float(denoise.detach()),following_loss=float(follow.detach()),groups=groups)

def run(assets_file,config,out,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);source=Path(config["source_study"])/"g2"
    public=read(Path(config["source_study"])/"public_summary.json")
    for name,expected in public["evidence"].items():
        if name.startswith("g2/") and sha(Path(config["source_study"])/name)!=expected:raise ValueError("Frozen G2 evidence differs")
    old=read(source/"protocol.json");assets=read(assets_file);manifest=metadata(assets)
    index={r["window_uid"]:r for r in manifest["rows"] if r["role"] in ("train","val")}
    chosen=old["evaluation_uids"];contrasts=old["contrast_uids"]
    if len(chosen)!=config["g2_music_cases"]:raise ValueError("Frozen case population changed")
    states={};weights={}
    for arm in config["g2_models"]:
        if arm=="g2_0":continue
        file=source/arm/"adapter.pt"
        if sha(file)!=public["model_hashes"][arm]:raise ValueError("Frozen G2 weights differ")
        saved=torch.load(file,map_location="cuda",weights_only=True)
        if saved["visibility_policy"]!="observed" or saved["protocol_sha256"]!=sha(source/"protocol.json"):raise ValueError("Saved G2 conditioning differs")
        states[arm]=saved["state"];weights[arm]=ref(file)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    def forbid(*args,**kwargs):raise RuntimeError("Attribution and replay optimizer forbidden")
    torch.optim.AdamW=forbid;torch.optim.Adam=forbid;torch.optim.SGD=forbid
    rt=AuditRuntime(assets,index,old["initial_adapter"],config["old_event_source"]);states["g2_0"]=copy.deepcopy(rt.initial)
    states={arm:states[arm] for arm in config["g2_models"]}
    tensors={};latents={};teacher=[];path=[];grads=[];generations=[];preferences=[];legacy_verified=0;old_denoise=dict(np.load(source/"denoising.npz"))
    times=list(reversed(torch.linspace(-1,999,51).int().tolist()));selected_times={times[s] for s in config["g2_sampler_steps"]}
    forward_calls=[0]
    def prehook(module,args,kwargs):
        forward_calls[0]+=1
        if rt.state.get("collect_latent") and kwargs.get("cond_drop_prob")==0 and int(args[2][0]) in selected_times:
            key=rt.state["trajectory_label"]+"_"+str(int(args[2][0]));latents[key]=args[0].detach().cpu().numpy().copy()
    handle=rt.model.register_forward_pre_hook(prehook,with_kwargs=True)
    protocol=dict(schema="g2_frozen_failure_attribution",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config=config,source_protocol=ref(source/"protocol.json"),weights=weights,initial_adapter=old["initial_adapter"],
        base=assets["checkpoint"],scaler=assets["scaler"],historical_cases_not_independent_blind=True,
        teacher_train_timestep_interval=[50,949],actual_sampler_times=times[:-1],optimizer_calls=0,
        changed_weights=False,changed_sampling=False,test_tensor_reads=0)
    write(out/"protocol.json",protocol)
    try:
        for case,(uid,other) in enumerate(zip(chosen,contrasts)):
            target,music,_,_,_,_=rt.batch_self([uid])
            for timestep in config["g2_timesteps"]:
                generator=torch.Generator(device="cuda").manual_seed(config["seed"]+case*1000+timestep)
                t=torch.tensor([timestep],device="cuda");noise=torch.randn(target.shape,device="cuda",generator=generator)
                x=rt.diffusion.q_sample(target,t,noise)
                for arm,state in states.items():
                    label="teacher_"+str(case)+"_"+str(timestep)+"_"+arm
                    rec=exchange(rt,uid,other,state,x,music,t,label,tensors);rec.update(case=case,arm=arm,context="target_q_sample")
                    teacher.append(rec)
                    if timestep in (100,500,900):
                        for control,s in (("A",0),("B",1)):
                            if not np.array_equal(tensors[label+"_"+control],old_denoise[str(case)+"_"+str(timestep)+"_"+arm+"_"+str(s)]):raise ValueError("Historical teacher prediction replay differs")
                            legacy_verified+=1
                    if arm=="g2_2":
                        g=gradients(rt,uid,x,music,t);g["case"]=case;grads.append(g)
            for seed in config["generation_seeds"]:
                folder=out/("case_"+str(case)+"_seed_"+str(seed));folder.mkdir();outputs={};metrics={};hashes=[]
                controls=[("bypass",None,None),("g2_0_0","g2_0",0),("g2_0_1","g2_0",1),("g2_2_0","g2_2",0),("g2_2_1","g2_2",1),("hard_0",None,0),("hard_1",None,1),("null","g2_2",None)]
                for name,arm,s in controls:
                    if arm:rt.adapter.load_state_dict(states[arm])
                    rt.adapter.eval();rt.adapter.requires_grad_(False)
                    rt.batch_self([uid],[(uid,other)[s or 0]])
                    rt.state.update(capture=False,enabled=arm is not None,collect_latent=name=="bypass",trajectory_label=str(case)+"_"+str(seed))
                    if name=="null":rt.state["geometry"]=torch.zeros_like(rt.state["geometry"])
                    values=torch.zeros_like(target);mask=torch.zeros_like(target,dtype=torch.bool)
                    hard=name.startswith("hard_")
                    if hard:
                        original=torch.from_numpy(rt.self_data((uid,other)[s])[0])[None].cuda()
                        values,mask=hard_keyframe_inputs(original,rt.source((uid,other)[s])[1].cuda())
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    with torch.no_grad():sample,sampler=constrained_ddim(rt.diffusion,music,values,mask,guidance="hard" if hard else "none")
                    rt.state["collect_latent"]=False
                    motion,m=physical(rt,sample,music,folder/name,assets);outputs[name]=motion;metrics[name]=m;hashes.append(sampler["noise_sha256"])
                    if name!="null":
                        old_folder=source/folder.name/name
                        for file in ("normalized.npy","native.npy","native_fk.npy"):
                            if not np.array_equal(np.load(old_folder/file),np.load(folder/name/file)):raise ValueError("Historical free generation differs: "+name+"/"+file)
                        legacy_verified+=1
                    generations.append(dict(case=case,seed=seed,control=name,**m,sampler=sampler))
                    print("G2_ATTR_GENERATED",case,seed,name,flush=True)
                if len(set(hashes))!=1 or not np.array_equal(outputs["null"],outputs["bypass"]):raise ValueError("Null/source visibility or noise invariant failed")
                refs=[rt.self_data(v)[2] for v in (uid,other)];region=np.ones((150,8),bool)
                base_activity=metrics["bypass"]["dynamics"]["upper8_angular_speed_mean_radps"]
                for arm in ("g2_0","g2_2","hard"):
                    a,b=outputs[arm+"_0"],outputs[arm+"_1"]
                    pa=source_error(a,refs[1],region)-source_error(a,refs[0],region);pb=source_error(b,refs[0],region)-source_error(b,refs[1],region)
                    gain=(source_error(outputs["bypass"],refs[0],region)+source_error(outputs["bypass"],refs[1],region)-source_error(a,refs[0],region)-source_error(b,refs[1],region))/2
                    activity=min(metrics[arm+"_"+str(s)]["dynamics"]["upper8_angular_speed_mean_radps"] for s in (0,1))/max(base_activity,1e-12)
                    preferences.append(dict(case=case,seed=seed,arm=arm,source_a_preference=pa,source_b_preference=pb,gain=gain,activity=activity,
                        direction_A=source_error(b,refs[0],region)-source_error(a,refs[0],region),
                        direction_B=source_error(a,refs[1],region)-source_error(b,refs[1],region),
                        old_following_pass=pa>.001 and pb>.001 and gain>0 and activity>=.9))
                for timestep in sorted(selected_times):
                    key=str(case)+"_"+str(seed)+"_"+str(timestep);x=torch.from_numpy(latents[key]).cuda();t=torch.tensor([timestep],device="cuda")
                    for arm,state in states.items():
                        label="path_"+key+"_"+arm
                        rec=exchange(rt,uid,other,state,x,music,t,label,tensors);rec.update(case=case,seed=seed,arm=arm,context="same_bypass_trajectory_latent")
                        path.append(rec)
        result=dict(schema="g2_attribution_result",teacher=teacher,common_state_path=path,gradients=grads,preferences=preferences,
            generations=len(generations),native_evaluations=len(generations),native_passes=sum(r["native_ok"] for r in generations),
            generation_records=generations,legacy_predictions_and_generations_exactly_replayed=legacy_verified,
            model_forward_calls=forward_calls[0],teacher_training_support=[50,949],untrained_sampler_timesteps=[t for t in times[:-1] if t<50 or t>949],
            null_equals_bypass=True,optimizer_calls=0,statistics_fit_calls=0,g3_started=False,operational_qualified=False)
        write(out/"result.json",result);np.savez_compressed(out/"traces.npz",**tensors);np.savez_compressed(out/"latents.npz",**latents)
        if replay_from:
            original=Path(replay_from)
            if read(original/"result.json")!=result:raise ValueError("G2 fresh attribution readouts differ")
            fields=0
            for name in ("traces.npz","latents.npz"):
                x=dict(np.load(original/name));y=dict(np.load(out/name))
                if set(x)!=set(y) or any(not np.array_equal(x[k],y[k]) for k in x):raise ValueError("G2 fresh trace/latent tensors differ")
                fields+=len(x)
            arrays=0
            for p in original.glob("case_*/*/normalized.npy"):
                for name in ("normalized.npy","native.npy","native_fk.npy"):
                    if not np.array_equal(np.load(p.parent/name),np.load(out/p.parent.relative_to(original)/name)):raise ValueError("G2 regenerated motion differs")
                    arrays+=1
            write(out/"replay.json",dict(status="FRESH_G2_ATTRIBUTION_AND_GENERATION_REPLAY",optimizer_calls=0,arrays_verified=arrays,trace_fields_verified=fields,regenerated=len(generations)))
    finally:handle.remove();rt.close()
    print("G2_ATTRIBUTION_COMPLETED",flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ("assets","config","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(a.assets,read(a.config),a.out,a.replay_from)
if __name__=="__main__":main()

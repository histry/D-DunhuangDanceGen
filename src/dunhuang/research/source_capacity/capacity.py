"""Frozen source trajectories, input-output derivatives and train-only site probes."""
import argparse,contextlib
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter
from dunhuang.generation.targets import digest
from dunhuang.generation.event_geometry import _matrix
from dunhuang.generation.source_following import following_loss
from dunhuang.research.source_tasks.math import cf_loss,angles,response
from dunhuang.research.source_identifiability.g2 import Runtime,physical,source_error
from dunhuang.research.transfer_audit.instrument import ForwardObserver
from dunhuang.sampling import constrained_ddim
from .common import read,commit,fresh,forbid_optimizers,same_arrays,frozen,weight
from .numerics import check_derivatives,trusted,cosine,trajectory_entry

def rms(value):return float(np.sqrt(np.square(np.asarray(value,dtype=float)).mean()))

def native(rt,pred):
    return MotionAdapter.to_native(rt.normalizer.unnormalize(pred.detach().cpu().clone())[0].numpy())

class AuditRuntime(Runtime):
    def inject(self,module,args,music):
        if not self.state.get("enabled",True):return music
        delta,reconstruction,trace=self.adapter(self.state["geometry"],music,return_aux=True,return_trace=True,visibility_policy="observed")
        self.state["reconstruction"]=reconstruction;self.state["residual"]=delta
        if self.state.get("capture"):
            self.trace.update({k:v.detach().cpu().numpy()[0] for k,v in trace.items()})
            self.trace["base_condition"]=music.detach().cpu().numpy()[0]
        return music+delta

@contextlib.contextmanager
def injection(rt,site,value):
    module=rt.model.cond_encoder if site=="cond" else rt.model.seqTransDecoder.stack[int(site.split("_")[1])]
    def add(module,args,output):
        if output.shape!=value.shape:raise ValueError("Matched 512D layer shape differs")
        return output+value
    handle=module.register_forward_hook(add)
    try:yield
    finally:handle.remove()

def project(rt,pred,mask):
    raw=(pred-torch.as_tensor(rt.bias,device="cuda"))/torch.as_tensor(rt.scale,device="cuda")
    r=_matrix(raw[...,79:127].reshape(-1,150,8,6)).transpose(-1,-2)
    return r[0][mask].flatten()

def exchange(rt,uids,x,music,t,states,arm,label,stored):
    rt.adapter.load_state_dict(states[arm]);rt.adapter.eval();rt.adapter.requires_grad_(False)
    outputs=[];traces=[]
    for s in range(2):
        rt.batch_self([uids[0]],[uids[s]]);rt.state["capture"]=True;rt.trace={}
        with torch.no_grad():pred=rt.model(x,music,t,cond_drop_prob=0.)
        outputs.append(pred.detach());traces.append({k:v.copy() for k,v in rt.trace.items()})
    rt.state["capture"]=False
    deltas={k:traces[1][k]-traces[0][k] for k in traces[0]}
    for name,value in deltas.items():stored[label+"__"+name]=value
    a,b=[native(rt,v) for v in outputs];refs=[rt.self_data(u)[2] for u in uids]
    unconditional=[]
    with torch.no_grad():
        for source in uids:
            rt.batch_self([uids[0]],[source]);unconditional.append(rt.model(x,music,t,cond_drop_prob=1.))
    if not torch.equal(unconditional[0],unconditional[1]):raise ValueError("Unconditional source invariance changed")
    guided=[unconditional[0]+2*(v-unconditional[0]) for v in outputs];clipped=[v.clamp(-1,1) for v in guided]
    known=rt.source(uids[0])[0].known[0,:,79:127].reshape(150,8,6).all(-1).numpy()&rt.source(uids[1])[0].known[0,:,79:127].reshape(150,8,6).all(-1).numpy()
    return dict(timestep=int(t.item()),arm=arm,layer_difference_rms={k:rms(v) for k,v in deltas.items()},
        source_response=response(a,b,refs[0],refs[1],known),
        guided_response=response(native(rt,guided[0]),native(rt,guided[1]),refs[0],refs[1],known),
        clipped_response=response(native(rt,clipped[0]),native(rt,clipped[1]),refs[0],refs[1],known),
        clipping_retention=float((clipped[1]-clipped[0]).norm()/(guided[1]-guided[0]).norm().clamp_min(1e-12)),
        unconditional_source_invariant=True,
        source_a_preference=source_error(a,refs[1],known)-source_error(a,refs[0],known),
        source_b_preference=source_error(b,refs[0],known)-source_error(b,refs[1],known)),deltas["residual_512"],traces[0]["base_condition"]

def probes(rt,uids,x,music,t,states,sites,config,label,stored):
    rt.adapter.load_state_dict(states["C2"]);rt.adapter.eval();rt.adapter.requires_grad_(False)
    base,delta,base_condition=exchange(rt,uids,x,music,t,states,"C2",label+"__exchange",stored)
    rt.batch_self([uids[0]],[uids[0]]);rt.state["capture"]=False
    refa,refb=[rt.self_data(u)[2] for u in uids]
    mask=rt.source(uids[0])[0].known[0,:,79:127].reshape(150,8,6).all(-1)&rt.source(uids[1])[0].known[0,:,79:127].reshape(150,8,6).all(-1)
    mask=mask.cuda();rr=torch.from_numpy(np.stack([refa,refb])).cuda()
    rotations=_matrix(rr[...,79:127].reshape(2,150,8,6))
    target=(rotations[1]-rotations[0])[mask].flatten();target=target/target.norm().clamp_min(1e-12)
    raw=torch.from_numpy(delta[None]).cuda();desired=config["probe"]["relative_rms"]*rms(base_condition)
    if raw.norm()==0:return [dict(status="NO_SOURCE_DIRECTION",site=s) for s in sites]
    records=[]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        for site in sites:
            gated=site.endswith("_gated");where=site.replace("_gated","")
            direction=raw.clone()
            if gated:direction=direction*mask.any(-1)[None,:,None]
            direction=direction*(desired/direction.square().mean().sqrt().clamp_min(1e-12))
            zero=torch.zeros_like(direction)
            def f(z):
                with injection(rt,where,z):
                    return project(rt,rt.model(x,music,t,cond_drop_prob=0.),mask)
            try:
                jvp,vjp,check=check_derivatives(f,zero,direction,target,config["probe"]["fd_steps"])
            except RuntimeError as exc:
                if not any(word in str(exc).lower() for word in ("derivative","backward","forward ad")):raise
                records.append(dict(site=site,status="AUTODIFF_UNQUALIFIED",numeric_trusted=False,error_type=type(exc).__name__))
                continue
            if not torch.isfinite(jvp).all() or not torch.isfinite(vjp).all():
                records.append(dict(site=site,status="NONFINITE_DERIVATIVE",numeric_trusted=False));continue
            with torch.no_grad():
                oracle=vjp*(direction.norm()/vjp.norm().clamp_min(1e-12))
                generator=torch.Generator(device="cuda").manual_seed(config["seed"]+int(t.item()))
                random=torch.randn(direction.shape,device="cuda",generator=generator);random*=direction.norm()/random.norm().clamp_min(1e-12)
                controls={}
                for kind,d in (("learned",direction),("oracle_target_dependent",oracle),("random",random)):
                    samples=[]
                    for sign in (-1,1):
                        with injection(rt,where,sign*d):samples.append(native(rt,rt.model(x,music,t,cond_drop_prob=0.)))
                    a,b=samples;region=mask.cpu().numpy()
                    controls[kind]=dict(toward_B=source_error(a,refb,region)-source_error(b,refb,region),
                        toward_A=source_error(b,refa,region)-source_error(a,refa,region),
                        response=response(a,b,refa,refb,region))
            check.update(site=site,status="MEASURED",numeric_trusted=trusted(check,config["probe"]),actual_source_delta_rms=rms(delta),
                probe_rms=desired,controls=controls,oracle_is_not_deployable=True,coordinates="rotation matrix chord; relative protocol diagnostic only")
            records.append(check)
            stored[label+"__"+site+"__jvp"]=jvp.cpu().numpy();stored[label+"__"+site+"__vjp"]=vjp.cpu().numpy()
    return records

def gradient_audit(rt,pair,x,music,t,masks,cfg):
    uids=[pair["a"],pair["b"]];target,music,geometry,reference,observed,_=rt.batch_self(uids)
    rt.adapter.requires_grad_(True);rt.adapter.eval()
    noisy=x.expand(2,-1,-1);times=t.expand(2)
    y=rt.model(noisy,music,times,cond_drop_prob=0.)
    keys=torch.from_numpy(np.stack([masks[u+"__expanded"] for u in uids])).cuda()
    mask=torch.from_numpy(masks[uids[0]+"__known"]&masks[uids[1]+"__known"]).cuda()
    scale=torch.as_tensor(rt.scale,device="cuda");bias=torch.as_tensor(rt.bias,device="cuda")
    losses=dict(mse=(y-target).square().mean(),following=following_loss(y,reference,observed,keys,scale,bias,"same_recording_pose_exact"),
        cf=cf_loss(y,reference,mask,scale,bias,cfg)[0])
    named=list(rt.adapter.named_parameters());parameters=[p for _,p in named];grads={}
    for name,loss in losses.items():grads[name]=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    groups={}
    for group,prefixes in (("output",("output",)),("source_encoder",("input","encoder")),("fusion",("music","attention","norm"))):
        vectors={}
        for loss,values in grads.items():
            vectors[loss]=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for (n,p),g in zip(named,values) if n.split(".")[0] in prefixes]).detach().cpu().numpy()
        groups[group]=dict(norms={k:float(np.linalg.norm(v)) for k,v in vectors.items()},
            cosines={a+"_"+b:cosine(vectors[a],vectors[b]) for a,b in (("mse","following"),("mse","cf"),("following","cf"))})
    rt.adapter.requires_grad_(False);rt.adapter.zero_grad(set_to_none=True)
    return dict(timestep=int(t.item()),groups=groups,losses={k:float(v.detach()) for k,v in losses.items()},
        inside_registered_cf_support=int(t.item()) in pair["ddim_support"],optimizer_calls=0)

def gate(records,cfg):
    sites=sorted({r["site"] for r in records});result={};improved={}
    baseline={(r["case"],r["timestep"]):r for r in records if r["site"]=="cond"}
    for site in sites:
        selected=[r for r in records if r["site"]==site]
        result[site]=bool(selected) and all(r.get("numeric_trusted",False) and r["controls"]["learned"]["toward_A"]>cfg["min_direction_rad"] and r["controls"]["learned"]["toward_B"]>cfg["min_direction_rad"] and (r["direction_cosine"] or -1)>cfg["min_direction_cosine"] for r in selected)
        improved[site]=site!="cond" and result[site] and all(baseline[(r["case"],r["timestep"])].get("numeric_trusted",False) for r in selected) and all(
            r["controls"]["learned"]["response"]["normalized_response"]>=cfg["min_relative_gain"]*baseline[(r["case"],r["timestep"])]["controls"]["learned"]["response"]["normalized_response"]
            for r in selected)
    return dict(train_directional_probe_pass=result,train_improved_sites=improved,optimizer_updates=0,
        lightweight_training_status="NOT_REGISTERED_WITHOUT_JOINT_NATIVE_AND_SOURCE_QUALIFICATION",
        local_probe_is_not_free_generation=True,phase_identity_equals_gating=True,operational_qualified=False)

def run(config,out,replay_from=None):
    out=fresh(out);root,pub,p,m=frozen(config);assets=read(checked(p["assets"]));cfg=read(root/"cf/protocol.json")["config"]["g2"]
    masks=dict(np.load(checked(p["supervision"])));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);forbid_optimizers()
    rt=AuditRuntime(assets,{r["window_uid"]:r for r in m["rows"]},m["initial_adapter"],m["audio_contract"])
    states={};weights={}
    for arm in config["arms"]:
        file=weight(root,pub,"cf",arm);payload=torch.load(file,map_location="cuda",weights_only=True)
        if payload["visibility_policy"]!="observed" or payload["protocol_sha256"]!=sha(root/"cf/protocol.json"):raise ValueError("Frozen CF policy differs")
        states[arm]=payload["state"];weights[arm]=ref(file)
    stored={};latents={};trajectories=[];layer_records=[];val_probes=[];train_probes=[];gradients=[];generations=[];predictions={}
    active={}
    def capture(module,args,kwargs):
        if active and kwargs.get("cond_drop_prob")==0:
            tick=int(args[2][0])
            if active["save_latent"] and tick in config["times"]:latents[str(active["case"])+"__"+str(tick)]=args[0].detach().cpu().numpy()
    observer=ForwardObserver(rt.model,capture)
    try:
        for case,pair in enumerate(m["validation_pairs"]):
            uids=[pair["a"],pair["b"]];target,music,*_=rt.batch_self([uids[0]])
            refs=[rt.self_data(u)[2] for u in uids];known=masks[uids[0]+"__known"]&masks[uids[1]+"__known"]
            folder=out/("case_"+str(case)+"_seed_"+str(config["generation_seed"]));folder.mkdir()
            for arm in config["arms"]:
                rt.adapter.load_state_dict(states[arm]);rt.adapter.eval();rt.adapter.requires_grad_(False)
                for s in (0,1):
                    rt.batch_self([uids[0]],[uids[s]]);active.update(case=case,save_latent=arm=="C2" and s==0)
                    sequence=[]
                    def record(step,t,pred,clean):sequence.append(pred.detach().cpu().numpy()[0])
                    torch.manual_seed(config["generation_seed"]);torch.cuda.manual_seed_all(config["generation_seed"])
                    values=torch.zeros_like(target);mask=torch.zeros_like(target,dtype=torch.bool)
                    sample,sampler=constrained_ddim(rt.diffusion,music,values,mask,guidance="none",observer=record)
                    active.clear();name=arm+"_"+str(s);motion,metrics=physical(rt,sample,music,folder/name,assets)
                    old=root/"cf"/folder.name/name
                    for f in ("normalized.npy","native.npy","native_fk.npy"):
                        if not np.array_equal(np.load(old/f),np.load(folder/name/f)):raise ValueError("Observation changed frozen generation")
                    if read(old/"physical.json")!=read(folder/name/"physical.json"):raise ValueError("Frozen Native differs")
                    predictions[(case,arm,s)]=sequence
                    stored[str(case)+"__"+arm+"__"+str(s)+"__trajectory"]=np.stack(sequence)
                    generations.append(dict(case=case,arm=arm,source=s,**metrics,sampler=sampler));print("CAPACITY_REGENERATED",case,name,flush=True)
                aseq,bseq=[predictions[(case,arm,s)] for s in (0,1)]
                for step,(a,b) in enumerate(zip(aseq,bseq)):
                    a=native(rt,torch.from_numpy(a[None]).cuda());b=native(rt,torch.from_numpy(b[None]).cuda())
                    trajectories.append(trajectory_entry(case,arm,step,a,b,refs[0],refs[1],known))
            for tick in config["times"]:
                x=torch.from_numpy(latents[str(case)+"__"+str(tick)]).cuda();t=torch.tensor([tick],device="cuda")
                for arm in config["arms"]:
                    rec,_,_=exchange(rt,uids,x,music,t,states,arm,str(case)+"__"+arm+"__"+str(tick),stored);rec["case"]=case;layer_records.append(rec)
                for rec in probes(rt,uids,x,music,t,states,["cond"],config,"val_"+str(case)+"_"+str(tick),stored):
                    rec.update(case=case,timestep=tick,scope="fixed_exposed_val_common_C2A_free_state");val_probes.append(rec)
        q=read(root/"pairs/protocol.json");pairs=read(checked(q["pairs"]))["train"]
        from dunhuang.data.split_registry import stable
        by_music={}
        for pair in sorted(pairs,key=lambda v:stable([config["seed"],v["a"],v["b"]])):by_music.setdefault(pair["music_uid"],pair)
        chosen=[by_music[k] for k in sorted(by_music,key=lambda k:stable([config["seed"],k]))[:config["probe"]["train_music_cases"]]]
        if len(chosen)!=config["probe"]["train_music_cases"]:raise ValueError("Fixed train diagnostic population insufficient")
        for case,pair in enumerate(chosen):
            uids=[pair["a"],pair["b"]];target,music,*_=rt.batch_self([uids[0]])
            generator=torch.Generator(device="cuda").manual_seed(config["seed"]+case+500)
            noise=torch.randn(target.shape,device="cuda",generator=generator)
            for tick in config["times"]:
                t=torch.tensor([tick],device="cuda");x=rt.diffusion.q_sample(target,t,noise)
                sites=config["probe"]["sites"]+["decoder_3_gated"]
                for rec in probes(rt,uids,x,music,t,states,sites,config,"train_"+str(case)+"_"+str(tick),stored):
                    rec.update(case=case,timestep=tick,scope="fixed_train_q_of_own_A");train_probes.append(rec)
                rt.adapter.load_state_dict(states["C2"]);grad=gradient_audit(rt,pair,x,music,t,masks,cfg);grad["case"]=case;gradients.append(grad)
        result=dict(generations=len(generations),native_passes=sum(r["native_ok"] for r in generations),native_evaluations=len(generations),
            generation_records=generations,trajectories=trajectories,layers=layer_records,val_derivatives=val_probes,train_site_derivatives=train_probes,
            gradient_conflicts=gradients,site_gate=gate(train_probes,config["probe"]),optimizer_calls=0,statistics_fit_calls=0,
            source_following_qualified=False,operational_qualified=False,g3_started=False)
        np.savez_compressed(out/"audit_arrays.npz",**stored,**{"latent__"+k:v for k,v in latents.items()})
        write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=config,weights=weights,
            source_protocol=ref(root/"cf/protocol.json"),preflight=ref(Path(p["manifest"]["path"]).parent/"protocol.json"),
            chosen_train_pairs=chosen,common_states="actual C2-A free trajectory for Val; q(X_A) for Train probes",
            oracle_uses_true_targets=True,identity_phase_equals_gated=True,changed_parameters=False,optimizer_calls=0))
        if replay_from:
            old=Path(replay_from)
            if read(old/"result.json")!=result:raise ValueError("Fresh capacity readouts differ")
            same_arrays(old/"audit_arrays.npz",out/"audit_arrays.npz")
            count=0
            for file in old.glob("case_*/*/native.npy"):
                for name in ("normalized.npy","native.npy","native_fk.npy"):
                    if not np.array_equal(np.load(file.parent/name),np.load(out/file.parent.relative_to(old)/name)):raise ValueError("Capacity regenerated motion differs")
                    count+=1
            write(out/"replay.json",dict(status="FRESH_CAPACITY_DERIVATIVE_AND_TRAJECTORY_REPLAY",optimizer_calls=0,regenerated=len(generations),arrays_verified=count))
        print("CAPACITY_COMPLETED",result["site_gate"],flush=True)
    finally:
        observer.close();rt.adapter.load_state_dict(states["C0"])
        # Original checkpoints are never written; frozen backbone is checked by close().
        rt.adapter.requires_grad_(False);rt.adapter.zero_grad(set_to_none=True);rt.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

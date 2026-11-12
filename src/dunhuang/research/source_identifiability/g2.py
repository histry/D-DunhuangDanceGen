"""Pre-registered G2: same architecture, exact-source supervision, frozen EDGE."""
import argparse,copy,json,subprocess,os
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.contracts import MotionAdapter
from dunhuang.generation.targets import phase_target,digest
from dunhuang.generation.event_music_adapter import authorize_event_adapter
from dunhuang.generation.source_following import consume,following_loss,interpolate_observed,hard_keyframe_inputs
from dunhuang.generation.event_geometry import event_geometry
from dunhuang.planning.primitive_contract import build
from dunhuang.alignment.clock_data import metadata
from dunhuang.alignment.conditional_candidates import validate_roles
from dunhuang.alignment.audio_clock import music_seconds
from dunhuang.alignment.sampling import group_schedule,coverage
from dunhuang.research.legacy.event_transfer import TransferRuntime
from dunhuang.research.mechanism import assembled
from dunhuang.research.event_causal_audit import layer_delta,tensor_hash
from dunhuang.research.event_metrics import dynamics,event_timing
from dunhuang.sampling import constrained_ddim

def read(p):return json.loads(Path(p).read_text())

def make_condition(row,event_source):
    event=dict(event_uid="self_reference",source_uid=row["window_uid"],reference=row["motion"],source_interval=[0,150],music_interval=[0,150],
        time_warp=[[0,0],[150,150]],primitive=dict(source_frame=75,music_frame=75,joints=list(range(12,20)),radius=12),feasibility=dict(native="UNKNOWN"))
    plan=dict(schema="event_plan_v1",fps=30,frames=150,audio_sha256=row["audio"]["sha256"],time_scale_limits=[.75,1.33],events=[event],
        audio_window=dict(start_seconds=float(music_seconds(row)),frames=150,fps=30))
    sidecar=Path(event_source)/"data/sidecars"/(row["audio"]["sha256"]+".json")
    seal=read(Path(event_source)/"preflight.json")["files"]["data/sidecars/"+sidecar.name]
    if sha(sidecar)!=seal:raise ValueError("Audio-only event sidecar seal differs")
    beats=[e["frame"]-row["start_frame"]-round(30*row["audio_offset_seconds"]) for e in read(sidecar)["events"] if e["event_type"]=="beat"]
    beats=[int(t) for t in beats if 12<=t<138 and not 25<=t<40 and not 90<=t<115]
    contract=build(plan,{row["window_uid"]:row["observed"]},beat_frames=beats)
    condition,keys=consume(plan,contract,"completion_upper8")
    return condition,keys,plan,contract

class Runtime(TransferRuntime):
    def __init__(self,assets,index,initial,event_source):
        super().__init__(assets,index,"cross")
        saved=torch.load(checked(initial),map_location="cuda",weights_only=True)
        self.adapter.load_state_dict(saved["state"],strict=True)
        self.initial=copy.deepcopy(self.adapter.state_dict());self.event_source=event_source;self.conditions={}
        self.trace={};self.trace_handles=[]
        for i,module in enumerate(self.model.seqTransDecoder.stack):
            self.trace_handles.append(module.register_forward_hook(self.capture("decoder_"+str(i))))
    def capture(self,name):
        def hook(module,args,value):
            if self.state.get("capture"):self.trace[name]=value.detach().cpu().numpy()[0]
        return hook
    def inject(self,module,args,music):
        if not self.state.get("enabled",True):return music
        output=self.adapter(self.state["geometry"],music,return_aux=True,visibility_policy="observed",return_trace=self.state.get("capture",False))
        if self.state.get("capture",False):
            delta,reconstruction,trace=output
            self.trace.update({k:v.detach().cpu().numpy()[0] for k,v in trace.items()})
        else:delta,reconstruction=output
        self.state["reconstruction"]=reconstruction
        return music+delta
    def self_data(self,uid):
        pair=dict(target_uid=uid,source_uid=uid)
        return self.data(pair)
    def source(self,uid):
        if uid not in self.conditions:self.conditions[uid]=make_condition(self.index[uid],self.event_source)
        return self.conditions[uid]
    def batch_self(self,uids,donors=None):
        original=[self.self_data(uid) for uid in uids]
        selected=uids if donors is None else donors
        sources=[self.source(uid) for uid in selected]
        original_sources=[self.source(uid) for uid in uids]
        target=torch.from_numpy(np.stack([v[0] for v in original])).cuda()
        music=torch.from_numpy(np.stack([v[1] for v in original])).cuda()
        value=torch.cat([v[0].value for v in sources]).cuda();known=torch.cat([v[0].known for v in sources]).cuda();phase=torch.cat([v[0].phase for v in sources]).cuda()
        geometry=event_geometry(value,known,phase);self.state.update(geometry=geometry,enabled=True,capture=False)
        reference=torch.from_numpy(np.stack([v[2] for v in original])).cuda()
        observed=torch.from_numpy(np.stack([v[3] for v in original])).cuda()
        keys=torch.cat([v[1] for v in original_sources]).cuda()
        return target,music,geometry,reference,observed,keys
    def close(self):
        for h in self.trace_handles:h.remove()
        super().close()

def source_derangement(schedule,rows,seed):
    flat=np.asarray(schedule).reshape(-1)
    buckets={}
    for position,i in enumerate(flat):buckets.setdefault(rows[int(i)]["source_group"],[]).append(position)
    maximum=max(map(len,buckets.values()))
    if maximum>len(flat)/2:raise ValueError("Exact source-marginal derangement requires balanced independent groups")
    rng=np.random.default_rng(seed);order=np.concatenate([rng.permutation(buckets[k]) for k in sorted(buckets)])
    donors=np.empty_like(flat);donors[order]=flat[np.roll(order,maximum)]
    if sorted(map(int,flat))!=sorted(map(int,donors)):raise ValueError("Corruption altered source marginal")
    for a,b in zip(flat,donors):
        if rows[int(a)]["source_group"]==rows[int(b)]["source_group"] or rows[int(a)]["choreography_uid"]==rows[int(b)]["choreography_uid"]:raise ValueError("Corruption retained source identity association")
    return donors.reshape(np.asarray(schedule).shape)

def preflight(assets,index,out,config,initial,registered):
    rows=list(index.values());validate_roles(rows);validated={}
    def check(r):
        k=(r["path"],r["sha256"])
        if k not in validated:validated[k]=checked(r)
        return validated[k]
    with np.load(check(assets["scaler"])) as z:scale,bias=z["scale"],z["bias"]
    for i,r in enumerate(rows):
        check(r["raw_motion"]);check(r["audio"])
        proof=read(check(r["alignment_attestation"]))
        if proof["status"]!="verified" or proof["motion_sha256"]!=r["raw_motion"]["sha256"] or proof["audio_sha256"]!=r["audio"]["sha256"] or abs(proof["audio_offset_seconds"]-r["audio_offset_seconds"])>1e-8:raise ValueError("Source clock attestation differs")
        native=np.load(check(r["motion"]));observed=np.load(check(r["observed"]));supervised=np.load(check(r["supervision"]))
        if native.shape!=(150,151) or observed.dtype!=bool or not observed[:,79:127].all() or not supervised.all():raise ValueError("Observed exact-target support incomplete")
        if phase_target(native,scale,bias)[0] is None:raise ValueError("Target outside original scaler support")
        receipt=read(check(r["jukebox_receipt"]))
        if receipt["window_uid"]!=r["window_uid"] or receipt["features_sha256"]!=r["jukebox"]["sha256"] or receipt["audio_sha256"]!=r["audio"]["sha256"] or receipt["layer"]!=66 or receipt["fps"]!=30 or receipt["silence_padding"]:raise ValueError("Real aligned Jukebox provenance differs")
        juke=np.load(check(r["jukebox"]),mmap_mode="r")
        if juke.shape!=(150,4800) or juke.dtype!=np.float32 or not np.isfinite(juke).all():raise ValueError("Jukebox tensor differs")
        if abs(receipt["sample_start"]/receipt["sample_rate"]-float(music_seconds(r)))>1/receipt["sample_rate"]:raise ValueError("Audible clock differs from extraction")
        if i%500==0:print("G2_PREFLIGHT",i,len(rows),flush=True)
    chosen=[];seen=set()
    for pair in registered:
        uid=pair["target_uid"];r=index[uid]
        if r["role"]!="val":raise ValueError("Registered evaluation must be Val")
        if r["music_uid"] not in seen:chosen.append(uid);seen.add(r["music_uid"])
        if len(chosen)==config["music_cases"]:break
    if len(chosen)!=config["music_cases"]:raise ValueError("Registered music cases unavailable")
    contrasts=[next(p["target_uid"] for p in registered if index[p["target_uid"]]["choreography_uid"]!=index[uid]["choreography_uid"]) for uid in chosen]
    train=[r for r in rows if r["role"]=="train"]
    schedule=group_schedule(train,config["steps"],config["batch"],config["seed"])
    donors=source_derangement(schedule,train,config["seed"]+31)
    np.save(out/"training_schedule.npy",schedule);np.save(out/"donor_schedule.npy",donors)
    return train,chosen,contrasts,coverage(train,schedule)

def train(rt,train_rows,out,config):
    schedule=np.load(out/"training_schedule.npy");donors=np.load(out/"donor_schedule.npy")
    weights=config["original_loss_weights"]
    if weights!=dict(velocity=.1,rotation=.02,source_reconstruction=.05):raise ValueError("Original reconstruction objective changed")
    scale=torch.as_tensor(rt.scale,device="cuda");bias=torch.as_tensor(rt.bias,device="cuda")
    base_initial=digest(rt.adapter);states={"g2_0":copy.deepcopy(rt.initial)};initials=[]
    for arm in ("g2_1","g2_2","g2_3"):
        rt.adapter.load_state_dict(rt.initial);rt.adapter.requires_grad_(True);rt.adapter.train();initials.append(digest(rt.adapter))
        folder=out/arm;folder.mkdir();history=[]
        optimizer=torch.optim.AdamW(rt.adapter.parameters(),lr=config["learning_rate"])
        for step,ids in enumerate(schedule):
            uids=[train_rows[i]["window_uid"] for i in ids]
            wrong=[train_rows[i]["window_uid"] for i in donors[step]] if arm=="g2_3" else None
            target,music,geometry,reference,observed,keys=rt.batch_self(uids,wrong)
            generator=torch.Generator(device="cuda").manual_seed(config["seed"]+step)
            t=torch.randint(50,950,(len(ids),),device="cuda",generator=generator)
            noise=torch.randn(target.shape,device="cuda",generator=generator)
            y=rt.model(rt.diffusion.q_sample(target,t,noise),music,t,cond_drop_prob=0.)
            loss,components=rt.loss(target,y,geometry)
            follow=following_loss(y,reference,observed,keys,scale,bias,"same_recording_pose_exact")
            if arm!="g2_1":loss=loss+config["following_weight"]*follow
            if not torch.isfinite(loss):raise ValueError("G2 nonfinite objective")
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(rt.adapter.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError("G2 nonfinite gradients")
            optimizer.step()
            if (step+1)%50==0:
                history.append(dict(step=step+1,**components,following_so3_squared=float(follow.detach()),total=float(loss.detach()),gradient_norm=float(norm),timestep_counts=[int(((t>=a)&(t<b)).sum()) for a,b in ((50,350),(350,650),(650,950))]))
                write(folder/"history.json",history);print("G2_TRAIN",arm,step+1,float(loss.detach()),flush=True)
        state=copy.deepcopy(rt.adapter.state_dict());states[arm]=state
        torch.save(dict(state=state,mode="cross",condition_schema="native_event_music_v1",base_sha256=rt.assets["checkpoint"]["sha256"],
            visibility_policy="observed",condition_policy="completion_upper8",training_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            protocol_sha256=sha(out/"protocol.json"),selected_step=config["steps"],arm=arm,operational_qualified=False),folder/"adapter.pt")
    if len(set(initials))!=1 or initials[0]!=base_initial:raise ValueError("Unequal matched G2 initialization")
    return states

def source_error(output,reference,region):
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import matrices
    a=matrices(output[:,79:127].reshape(-1,8,6));b=matrices(reference[:,79:127].reshape(-1,8,6))
    angles=Rotation.from_matrix((a.swapaxes(-1,-2)@b).reshape(-1,3,3)).magnitude().reshape(150,8)
    return float(angles[region].mean())

def physical(rt,sample,music,folder,assets):
    folder.mkdir();assembled(rt,sample,music,folder)
    request=dict(assets,frames=150,output=str(folder),anchors=[],beat_frames=[])
    write(folder/"physical_request.json",request)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[3])+os.pathsep+os.environ.get("PYTHONPATH",""))
    subprocess.run([assets["native_python"],"-m","dunhuang.physical",str(folder/"physical_request.json")],env=env,check=True,stdout=subprocess.DEVNULL)
    motion=np.load(folder/"native.npy");fk=np.load(folder/"native_fk.npy");audit=read(folder/"physical.json")
    return motion,dict(native_ok=audit["absolute"]["ok"],native_reasons=audit["absolute"]["reasons"],dynamics=dynamics(motion,fk),
        continuous={k:float(v) for k,v in audit["audit"].items() if isinstance(v,(int,float))})

def evaluate(rt,states,index,chosen,contrasts,out,config):
    records=[];prefs=[];denoising=[];snapshots={};layers={};noise_hashes={};count=0
    for case,(uid,other) in enumerate(zip(chosen,contrasts)):
        target,music,_,_,_,_=rt.batch_self([uid]);refs=[rt.self_data(v)[2] for v in (uid,other)]
        conditions=[rt.source(v) for v in (uid,other)]
        for s,condition in enumerate(conditions):
            folder=out/("contract_"+str(case)+"_"+str(s));folder.mkdir();write(folder/"plan.json",condition[2]);write(folder/"condition.json",condition[3])
        for timestep in config["timesteps"]:
            generator=torch.Generator(device="cuda").manual_seed(config["seed"]+case*1000+timestep)
            t=torch.tensor([timestep],device="cuda");noisy=rt.diffusion.q_sample(target,t,torch.randn(target.shape,device="cuda",generator=generator))
            for arm,state in states.items():
                rt.adapter.load_state_dict(state);rt.adapter.eval();base=None;native_predictions=[]
                for s in range(2):
                    rt.batch_self([uid],[ (uid,other)[s] ]);rt.state["capture"]=True;rt.trace={}
                    with torch.no_grad():y=rt.model(noisy,music,t,cond_drop_prob=0.)
                    key=str(case)+"_"+str(timestep)+"_"+arm+"_"+str(s);snapshots[key]=y.cpu().numpy()
                    native_predictions.append(MotionAdapter.to_native(rt.normalizer.unnormalize(y.detach().cpu().clone())[0].numpy()))
                    if base is None:base={k:v.copy() for k,v in rt.trace.items()}
                    else:
                        denoising.append(dict(case=case,timestep=timestep,arm=arm,noisy_sha256=tensor_hash(noisy),music_sha256=tensor_hash(music),source_exchange_layers={k:layer_delta(v,base[k]) for k,v in rt.trace.items()}))
                        for k,v in rt.trace.items():layers[key+"_"+k]=v-base[k]
                        region=np.ones((150,8),bool);ya,yb=native_predictions
                        denoising[-1]["source_a_preference"]=source_error(ya,refs[1],region)-source_error(ya,refs[0],region)
                        denoising[-1]["source_b_preference"]=source_error(yb,refs[0],region)-source_error(yb,refs[1],region)
                    rt.state["capture"]=False
        for seed in config["generation_seeds"]:
            folder=out/("case_"+str(case)+"_seed_"+str(seed));folder.mkdir()
            outputs={};noise=[]
            def generate(name,source_index,hard=False):
                nonlocal count
                if source_index is not None:rt.batch_self([uid],[(uid,other)[source_index]])
                values=torch.zeros_like(target);mask=torch.zeros_like(target,dtype=torch.bool)
                if hard:
                    rt.state["enabled"]=False
                    ref_target=torch.from_numpy(rt.self_data((uid,other)[source_index])[0])[None].cuda()
                    keys=conditions[source_index][1].cuda()
                    values,mask=hard_keyframe_inputs(ref_target,keys)
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad():sample,sampler=constrained_ddim(rt.diffusion,music,values,mask,guidance="hard" if hard else "none")
                motion,metrics=physical(rt,sample,music,folder/name,rt.assets)
                count+=1;noise.append(sampler["noise_sha256"]);outputs[name]=motion;noise_hashes[str(case)+"_"+str(seed)+"_"+name]=sampler["noise_sha256"]
                s=0 if source_index is None else source_index
                known=conditions[s][0].known[0,:,79:127].reshape(150,8,6).all(-1).numpy()
                keys=conditions[s][1][0].numpy()
                records.append(dict(case=case,seed=seed,control=name,**metrics,known_error_radians=source_error(motion,refs[s],known),
                    unprovided_error_radians=source_error(motion,refs[s],~known),key_error_radians=source_error(motion,refs[s],keys),
                    event_timing=event_timing(motion,refs[s]),sampler=sampler))
                print("G2_GENERATED",case,seed,name,flush=True)
            rt.state.update(enabled=False,capture=False);generate("bypass",None)
            for arm,state in states.items():
                rt.adapter.load_state_dict(state);rt.adapter.eval();rt.adapter.requires_grad_(False)
                for s in range(2):generate(arm+"_"+str(s),s)
            rt.state.update(enabled=False,capture=False)
            for s in range(2):generate("hard_"+str(s),s,True);rt.state["enabled"]=False
            for s in range(2):
                name="interpolation_"+str(s);condition=conditions[s][0]
                value=interpolate_observed(outputs["bypass"],refs[s],condition.known[0].numpy())
                normalized=(MotionAdapter.to_edge(value)*rt.scale+rt.bias).astype(np.float32)
                motion,metrics=physical(rt,torch.from_numpy(normalized)[None].cuda(),music,folder/name,rt.assets);outputs[name]=motion
                known=condition.known[0,:,79:127].reshape(150,8,6).all(-1).numpy()
                records.append(dict(case=case,seed=seed,control=name,**metrics,known_error_radians=source_error(motion,refs[s],known),unprovided_error_radians=source_error(motion,refs[s],~known),
                    key_error_radians=source_error(motion,refs[s],conditions[s][1][0].numpy()),event_timing=event_timing(motion,refs[s]),post_generation_pose_paste=True,learned=False))
            if len(set(noise))!=1:raise ValueError("Counterfactual sampler noise differs")
            bypass=outputs["bypass"];zero_activity=next(r["dynamics"]["upper8_angular_speed_mean_radps"] for r in records if r["case"]==case and r["seed"]==seed and r["control"]=="bypass")
            for arm in list(states)+["hard","interpolation"]:
                ya,yb=outputs[arm+"_0"],outputs[arm+"_1"];all_region=np.ones((150,8),bool)
                a=source_error(ya,refs[1],all_region)-source_error(ya,refs[0],all_region);b=source_error(yb,refs[0],all_region)-source_error(yb,refs[1],all_region)
                gain=(source_error(bypass,refs[0],all_region)+source_error(bypass,refs[1],all_region)-source_error(ya,refs[0],all_region)-source_error(yb,refs[1],all_region))/2
                activity=min(r["dynamics"]["upper8_angular_speed_mean_radps"] for r in records if r["case"]==case and r["seed"]==seed and r["control"] in (arm+"_0",arm+"_1"))/max(zero_activity,1e-12)
                prefs.append(dict(case=case,seed=seed,arm=arm,source_a_preference=a,source_b_preference=b,mean_preference=(a+b)/2,reference_gain_vs_bypass=gain,activity_retention=activity,
                    source_following_pass=a>.001 and b>.001 and gain>0 and activity>=.9))
    np.savez_compressed(out/"denoising.npz",**snapshots);np.savez_compressed(out/"exchange_layers.npz",**layers)
    return dict(generations=count,nonlearned_interpolations=len(config["generation_seeds"])*len(chosen)*2,
        records=records,preferences=prefs,denoising=denoising,noise_hashes=noise_hashes,
        arm_following_gate={arm:all(p["source_following_pass"] for p in prefs if p["arm"]==arm) for arm in states},
        native_passes=sum(r["native_ok"] for r in records),native_evaluations=len(records),new_blind_source=False,
        actual_joint_scope=list(range(12,20)),new_root_contact_or_wrist_control=False,
        g3_started=False,sqp_started=False,operational_qualified=False)

def validate_config(config):
    if config["condition_policy"]!="completion_upper8" or config["visibility_policy"]!="observed" or config["train_scope"]!="same_recording_pose_exact":
        raise ValueError("G2 registered conditioning/supervision differs")
    if config["original_loss_weights"]!=dict(velocity=.1,rotation=.02,source_reconstruction=.05):
        raise ValueError("G2 original reconstruction objective differs")
    if config["test_tensor_reads"]!=0 or config["selection"]!="fixed_last_no_validation_selection":
        raise ValueError("G2 data roles or checkpoint selection differs")
    if type(config["steps"]) is not int or config["steps"]<1 or config["following_weight"]<=0:
        raise ValueError("G2 positive registered training budget required")

def run(assets_file,out,config,root_protocol,replay_from=None):
    validate_config(config)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);assets=read(assets_file)
    manifest=metadata(assets);index={r["window_uid"]:r for r in manifest["rows"] if r["role"] in ("train","val")}
    initial=read(root_protocol)["initial_adapter"];saved=torch.load(checked(initial),map_location="cpu",weights_only=True)
    if saved["mode"]!="cross" or saved["condition_schema"]!="native_event_music_v1":raise ValueError("Frozen G2 baseline differs")
    authorize_event_adapter(saved,assets["checkpoint"]["sha256"],True)
    registered=read(Path(initial["path"]).parent/"protocol.json")["validation_pairs"]
    train_rows,chosen,contrasts,cov=preflight(assets,index,out,config,initial,registered)
    protocol=dict(schema="g2_source_identifiability_v1",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),config=config,
        initial_adapter=initial,paired_manifest=assets["paired_generation_manifest"],scaler=assets["scaler"],base=assets["checkpoint"],
        train_uids=[r["window_uid"] for r in train_rows],evaluation_uids=chosen,contrast_uids=contrasts,schedule_coverage=cov,
        selection="fixed_last_no_validation_selection",scope="same-recording exact training, real source-exchange evaluation",negative_control="exact source occurrence marginal preserved by derangement",test_tensor_reads=0,operational_qualified=False)
    write(out/"protocol.json",protocol);write(out/"preflight.json",dict(status="PASSED_REAL_SOURCE_JUKEBOX_PREFLIGHT",train=len(train_rows),val=len(index)-len(train_rows),test_tensor_reads=0))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    rt=Runtime(assets,index,initial,config["event_source"])
    protocol.update(adapter_parameters=sum(p.numel() for p in rt.adapter.parameters()),initial_parameter_digest=digest(rt.adapter),
        frozen_parameter_digest=rt.before,source_condition_consumer="canonical_primitive_contract_completion_upper8_observed")
    write(out/"protocol.json",protocol)
    try:
        if replay_from:
            def forbid(*args,**kwargs):raise RuntimeError("Replay optimizer forbidden")
            torch.optim.AdamW=forbid;torch.optim.Adam=forbid;torch.optim.SGD=forbid
            states={"g2_0":copy.deepcopy(rt.initial)}
            for arm in ("g2_1","g2_2","g2_3"):
                p=torch.load(Path(replay_from)/arm/"adapter.pt",map_location="cuda",weights_only=True)
                if p["protocol_sha256"]!=sha(Path(replay_from)/"protocol.json"):raise ValueError("Saved G2 protocol seal differs")
                states[arm]=p["state"]
            if any(not np.array_equal(np.load(Path(replay_from)/name),np.load(out/name)) for name in ("training_schedule.npy","donor_schedule.npy")):raise ValueError("G2 schedule replay differs")
        else:states=train(rt,train_rows,out,config)
        rt.adapter.requires_grad_(False);rt.adapter.zero_grad(set_to_none=True);rt.adapter.eval()
        result=evaluate(rt,states,index,chosen,contrasts,out,config)
        write(out/"result.json",result)
        if replay_from:
            original=Path(replay_from)
            for name in ("denoising","exchange_layers"):
                a=np.load(original/(name+".npz"));b=np.load(out/(name+".npz"))
                if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError("G2 fresh scientific tensor replay differs")
            if read(original/"result.json")!=result:raise ValueError("G2 final metrics replay differ")
            arrays=0
            for p in original.glob("case_*/**/native.npy"):
                folder=p.parent.relative_to(original)
                for name in ("native.npy","native_fk.npy","normalized.npy"):
                    if not np.array_equal(np.load(original/folder/name),np.load(out/folder/name)):raise ValueError("G2 generated output replay differs")
                    arrays+=1
                if read(original/folder/"physical.json")!=read(out/folder/"physical.json"):raise ValueError("G2 Native replay differs")
            write(out/"replay.json",dict(status="FRESH_G2_GENERATION_REPLAY_VERIFIED",optimizer_calls=0,arrays_verified=arrays,regenerated=result["generations"],interpolations_recomputed=result["nonlearned_interpolations"]))
        print("G2_COMPLETED",flush=True)
    finally:rt.close()

def main():
    p=argparse.ArgumentParser()
    for k in ("assets","out","config","root-protocol"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(a.assets,a.out,read(a.config),a.root_protocol,a.replay_from)
if __name__=="__main__":main()

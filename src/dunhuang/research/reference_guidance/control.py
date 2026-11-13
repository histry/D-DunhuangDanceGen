"""R4-0: inference-available reference directions, single predicted-x0 step only."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from dunhuang.generation.event_geometry import _matrix,event_geometry
from dunhuang.generation.source_following import geodesic_squared
from dunhuang.research.source_identifiability.g2 import Runtime,physical,source_error
from dunhuang.research.source_capacity.capacity import native,injection
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays,frozen,weight
from dunhuang.research.source_capacity.numerics import check_derivatives,trusted,normalized_supported
from .core import reference,protection

def control_norm(base_condition,relative_rms):
    return float(base_condition.detach().double().norm())*relative_rms

def reference_loss(rt,pred,safe,weights):
    raw=(pred-torch.as_tensor(rt.bias,device="cuda"))/torch.as_tensor(rt.scale,device="cuda")
    pr=_matrix(raw[...,79:127].reshape(1,150,8,6)).transpose(-1,-2)
    rr=_matrix(safe)
    return (geodesic_squared(pr[0],rr)*weights).sum()/weights.sum()

def directions(predict,zero,available,base_condition,adapter,geometry,config,seed):
    # No full target or alternative reference can enter this controller.
    source,visible,stage,weights=available
    safe=torch.from_numpy(source).cuda()
    known=torch.from_numpy(visible.astype(np.float32)).cuda()
    weighted=torch.from_numpy(weights.astype(np.float32)).cuda()
    full=torch.ones((1,150,1),device="cuda",dtype=torch.bool)
    gate=torch.from_numpy(stage.any(-1)[None,:,None]).cuda()
    desired=control_norm(base_condition,config["probe"]["relative_rms"])
    rt=predict.runtime
    objectives={"B1":known,"B2":weighted};result={};checks={}
    for name,w in objectives.items():
        support=gate if name=="B2" else full
        def f(z):return reference_loss(rt,predict(z),safe,w).reshape(1)
        with torch.enable_grad():
            z=zero.detach().clone().requires_grad_()
            grad=torch.autograd.grad(f(z).sum(),z)[0]
        direction=normalized_supported(-grad,support,desired)
        j,g,check=check_derivatives(f,zero,direction,torch.ones(1,device="cuda"),config["probe"]["fd_steps"])
        check.update(numeric_trusted=trusted(check,config["probe"]),zero_direction=bool(direction.norm()==0),reference_only=True)
        checks[name]=check;result[name]=direction.detach()
    with torch.no_grad():
        learned=adapter(geometry,base_condition,visibility_policy="observed")
        if isinstance(learned,tuple):learned=learned[0]
        result["B0"]=normalized_supported(learned,full,desired)
        rng=torch.Generator(device="cuda").manual_seed(seed)
        noise=torch.randn(zero.shape,device="cuda",generator=rng)
        result["B3"]=normalized_supported(noise,full,desired)
        result["B3_stage"]=normalized_supported(noise,gate,desired)
    return result,checks

def qualification(records,cfg):
    result={}
    for kind,random in (("B1","B3"),("B2","B3_stage")):
        selected=[r for r in records if r["scope"]=="train" and r["arm"]==kind]
        result[kind]=len(selected)==24 and len({(r["case"],r["timestep"],r["source"]) for r in selected})==24 and all(r["numeric_trusted"] and r["protection"]["protected"] and r["own_gain_rad"]>cfg["min_direction_rad"] and r["directional_gain_rad"]>cfg["min_direction_rad"] and r["own_gain_rad"]>next(v["own_gain_rad"] for v in records if v["scope"]=="train" and v["case"]==r["case"] and v["timestep"]==r["timestep"] and v["source"]==r["source"] and v["arm"]==random) for r in selected)
    return dict(train_all_cases_all_times_pass=result,full_chain_started=False,mapping_training_started=False,operational_qualified=False)

def run(config,out,replay_from=None):
    out=fresh(out);root,pub,p,manifest=frozen(config);capacity=Path(config["capacity"])
    seals=read("reports/source_capacity_summary.json");array_seals=read("reports/source_capacity_checks.json")["private_output_array_sha256"]
    for name in ("protocol.json","result.json"):
        if sha(capacity/"capacity"/name)!=seals["evidence"]["capacity/"+name]:raise ValueError("Capacity lineage differs")
    arrays_file=capacity/"capacity/audit_arrays.npz"
    if sha(arrays_file)!=array_seals["capacity/audit_arrays.npz"]:raise ValueError("Actual latent seal differs")
    latents=dict(np.load(arrays_file));cp=read(capacity/"capacity/protocol.json")
    assets=read(checked(p["assets"]));index={r["window_uid"]:r for r in manifest["rows"]}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);forbid_optimizers()
    rt=Runtime(assets,index,manifest["initial_adapter"],manifest["audio_contract"])
    file=weight(root,pub,"cf","C2");saved=torch.load(file,map_location="cuda",weights_only=True)
    if saved["visibility_policy"]!="observed" or saved["protocol_sha256"]!=sha(root/"cf/protocol.json"):raise ValueError("Frozen source interface differs")
    rt.adapter.load_state_dict(saved["state"]);rt.adapter.eval();rt.adapter.requires_grad_(False);before=digest(rt.adapter)
    records=[];derivatives=[];stored={};native_records=[];contracts=[]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    try:
        for scope,pairs in (("train",cp["chosen_train_pairs"]),("val",manifest["validation_pairs"])):
            for case,pair in enumerate(pairs):
                uids=[pair["a"],pair["b"]]
                if any(index[u]["role"]!=scope for u in uids):raise ValueError("Source population role differs")
                target,music,*_=rt.batch_self([uids[0]])
                noise=torch.randn(target.shape,device="cuda",generator=torch.Generator(device="cuda").manual_seed(config["seed"]+case+500))
                available=[];geometries=[];refs=[]
                for uid in uids:
                    condition,keys,plan,contract=rt.source(uid)
                    a=reference(condition.value[0].numpy(),condition.known[0].numpy(),keys[0].numpy(),config["probe"]["phase_radius"],config["probe"]["phase_weight"])
                    available.append(a);refs.append(rt.self_data(uid)[2])
                    # Geometry is reconstructed from only the same provided source.
                    geometries.append(event_geometry(condition.value.cuda(),condition.known.cuda(),condition.phase.cuda()))
                    contracts.append(dict(scope=scope,case=case,plan=plan,contract=contract))
                common=available[0][1]&available[1][1]
                if not common.any():raise ValueError("No common observed evaluation region")
                rt.state.update(enabled=False,capture=False)
                for tick in config["times"]:
                    t=torch.tensor([tick],device="cuda")
                    x=torch.from_numpy(latents["latent__"+str(case)+"__"+str(tick)]).cuda() if scope=="val" else rt.diffusion.q_sample(target,t,noise)
                    capture={}
                    def hook(module,args,value):capture["base"]=value.detach().clone()
                    h=rt.model.cond_encoder.register_forward_hook(hook)
                    with sdpa_kernel(SDPBackend.MATH),torch.no_grad():base_pred=rt.diffusion.model_predictions(x,music,t,weight=2,clip_x_start=True)[1]
                    h.remove();base_condition=capture["base"];base=native(rt,base_pred)
                    zero=torch.zeros_like(base_condition)
                    def predict(z):
                        with injection(rt,"cond",z):
                            return rt.diffusion.model_predictions(x,music,t,weight=2,clip_x_start=True)[1]
                    predict.runtime=rt
                    label=scope+"__"+str(case)+"__"+str(tick)
                    stored[label+"__base"]=base_pred.detach().cpu().numpy()
                    if scope=="val" and tick==19:
                        _,metric=physical(rt,base_pred,music,out/(label+"__base"),assets)
                        native_records.append(dict(case=case,arm="base",source=None,**metric))
                    with sdpa_kernel(SDPBackend.MATH):
                        # Recompute common baseline under the same math kernel used by all probes.
                        with torch.no_grad():base_pred=predict(zero);base=native(rt,base_pred)
                        stored[label+"__base"]=base_pred.detach().cpu().numpy()
                        for s in (0,1):
                            ds,checks=directions(predict,zero,available[s],base_condition,rt.adapter,geometries[s],config,config["seed"]+case*1000+tick)
                            for kind,check in checks.items():derivatives.append(dict(scope=scope,case=case,timestep=tick,source=s,arm=kind,**check))
                            with torch.no_grad():
                                for arm,d in ds.items():
                                    pred=predict(d);motion=native(rt,pred)
                                    own=source_error(base,refs[s],common)-source_error(motion,refs[s],common)
                                    other=source_error(base,refs[1-s],common)-source_error(motion,refs[1-s],common)
                                    hidden=~available[s][1]
                                    record=dict(scope=scope,case=case,timestep=tick,source=s,arm=arm,own_gain_rad=own,directional_gain_rad=own-other,
                                        own_error_rad=source_error(motion,refs[s],common),wrong_gain_rad=other,
                                        hidden_error_rad=source_error(motion,refs[s],hidden) if hidden.any() else None,
                                        numeric_trusted=checks[arm]["numeric_trusted"] if arm in checks else None,
                                        protection=protection(base,motion,config["protection"]),input_norm=float(d.norm()))
                                    records.append(record);key=label+"__"+str(s)+"__"+arm
                                    stored[key+"__direction"]=d.cpu().numpy();stored[key+"__predicted_x0"]=pred.cpu().numpy()
                                    if scope=="val" and tick==19:
                                        _,metric=physical(rt,pred,music,out/key,assets)
                                        native_records.append(dict(case=case,arm=arm,source=s,**metric))
                    print("REFERENCE_PROBE",scope,case,tick,flush=True)
        if digest(rt.adapter)!=before:raise ValueError("Frozen adapter changed")
    finally:rt.close()
    result=dict(records=records,derivatives=derivatives,qualification=qualification(records,config["probe"]),
        native_single_step_readouts=native_records,native_evaluations=len(native_records),native_passes=sum(r["native_ok"] for r in native_records),
        full_dances_generated=0,optimizer_calls=0,statistics_fit_calls=0,human_reviewed=False,operational_qualified=False)
    np.savez_compressed(out/"arrays.npz",**stored)
    write(out/"contracts.json",contracts);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,capacity_protocol=ref(capacity/"capacity/protocol.json"),latents=ref(arrays_file),adapter=ref(file),
        contracts=ref(out/"contracts.json"),source_protocol=ref(root/"cf/protocol.json"),scope="single-step clipped CFG2 predicted-x0, condition512 only",
        inference_reads="provided source upper8+known+legal keys and identity verified map; targets evaluation only",
        stage="radius2 around actual provided legal keys; no hidden derivatives",native_scope="Val t19 predicted-x0 only; not full sampling"))
    if replay_from:
        old=Path(replay_from)
        if read(old/"result.json")!=result:raise ValueError("Reference probe readouts differ")
        same_arrays(old/"arrays.npz",out/"arrays.npz")
        count=0
        for f in old.glob("*__*/native.npy"):
            for name in ("normalized.npy","native.npy","native_fk.npy"):
                if not np.array_equal(np.load(f.parent/name),np.load(out/f.parent.name/name)):raise ValueError("Reference Native array replay differs")
                count+=1
            if read(f.parent/"physical.json")!=read(out/f.parent.name/"physical.json"):raise ValueError("Reference physical replay differs")
        write(out/"replay.json",dict(status="FRESH_REFERENCE_CONTROL_REPLAY",optimizer_calls=0,native_arrays_verified=count,exact_arrays=True))
    print("REFERENCE_CONTROL_COMPLETED",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

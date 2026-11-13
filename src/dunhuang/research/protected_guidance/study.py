"""R5-P0: fixed small condition subspace, exact forward rejection, no sampling."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from dunhuang.generation.event_geometry import _matrix
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays,frozen,weight
from dunhuang.research.source_capacity.capacity import native,injection
from dunhuang.research.source_capacity.numerics import normalized_supported,check_derivatives,trusted
from dunhuang.research.reference_guidance.control import control_norm,reference_loss
from dunhuang.research.reference_guidance.core import phase_reference,protection
from dunhuang.research.source_identifiability.g2 import Runtime,physical,source_error
from .solver import basis,project,physical_coordinates

def frozen_reference(config):
    summary=read("reports/reference_guidance_summary.json");receipt=read("reports/reference_guidance_receipt.json")
    if sha("reports/reference_guidance_summary.json")!=receipt["summary_sha256"] or receipt["exit_status"]!=0:raise ValueError("Frozen reference receipt differs")
    control=Path(config["reference_control"])
    for n in ("protocol.json","result.json"):
        if sha(control/"control"/n)!=summary["evidence"]["control/"+n]:raise ValueError("Frozen control changed")
    if sha(control/"control/arrays.npz")!=summary["private_array_sha256"]["control/arrays.npz"]:raise ValueError("Frozen directions changed")
    if config["protection"]!=read(control/"control/protocol.json")["config"]["protection"]:raise ValueError("Protection bounds changed")
    return control,summary

def guided(predict,zero,rt,base_rotation,available,radius,cfg,kind):
    safe,visible,stage,weights=available
    support=torch.from_numpy(stage.any(-1)[None,:,None]).cuda() if kind=="B2" else torch.ones((1,150,1),device="cuda",dtype=torch.bool)
    w=torch.from_numpy((weights if kind=="B2" else visible).astype(np.float32)).cuda()
    safe=torch.from_numpy(safe).cuda()
    z=zero.detach().clone().requires_grad_()
    g=torch.autograd.grad(reference_loss(rt,predict(z),safe,w),z)[0]
    seed=normalized_supported(-g,support,radius).detach()
    if float(seed.norm())==0:return seed,seed,dict(status="NO_RESOLVED_SOURCE_DIRECTION",numeric_trusted=False)
    def f(z):return physical_coordinates(rt,predict(z),base_rotation)
    vectors,a,gram,target=basis(f,zero,seed,support,radius,cfg["rank"])
    coef,solve=project(a,gram,target,cfg)
    direction=sum((float(v)*b*radius for v,b in zip(coef,vectors)),torch.zeros_like(zero)).detach()
    if direction.norm()>radius*(1+1e-6):raise ValueError("Condition budget exceeded")
    _,j0=torch.autograd.functional.jvp(f,zero,seed)
    readout=j0.detach()/j0.norm().clamp_min(1e-12)
    jp,gp,check=check_derivatives(f,zero,direction,readout,cfg["fd_steps"])
    expected=torch.from_numpy(a@coef).to(jp)
    consistency=float((jp-expected).norm()/jp.norm().clamp_min(1e-12))
    solve.update(derivatives=check,numeric_trusted=trusted(check,cfg) and consistency<1e-4 and solve["status"]=="SOLVED_FINITE_SUBSPACE",
        linear_combination_relative_error=consistency,condition_fraction=float(direction.double().norm())/radius,
        matrix_free=True,dense_jacobian_built=False)
    return direction,seed,solve

def qualify(records,solver_records,cfg):
    out={}
    for arm in ("PB1","PB2"):
        rs=[r for r in records if r["scope"]=="train" and r["arm"]==arm]
        baseline="B"+arm[-1];random="random_"+arm
        def joint(r):return r["own_gain_rad"]>cfg["min_direction_rad"] and r["directional_gain_rad"]>cfg["min_direction_rad"] and r["protection"]["protected"]
        lookup={(r["scope"],r["case"],r["timestep"],r["source"],r["arm"]):r for r in records}
        matched=lambda r,name:lookup[(r["scope"],r["case"],r["timestep"],r["source"],name)]
        count=sum(joint(r) and r["numeric_trusted"] for r in rs)
        old=sum(joint(matched(r,baseline)) for r in rs)
        gate=len(rs)==24 and count==24 and all(r["own_gain_rad"]>matched(r,random)["own_gain_rad"] for r in rs)
        out[arm]=dict(train_joint_trusted=count,train_baseline_joint=old,train_denominator=len(rs),all_cases_pass=gate,
            improvement_over_baseline=count-old,no_global_infeasibility_claim=True)
    return out

def run(config,out,replay_from=None):
    out=fresh(out);old,oldpub=frozen_reference(config);old_arrays=dict(np.load(old/"control/arrays.npz"))
    root,pub,p,manifest=frozen(config);assets=read(checked(p["assets"]));index={r["window_uid"]:r for r in manifest["rows"]}
    cp=read(checked(read(old/"control/protocol.json")["capacity_protocol"]))
    latents=dict(np.load(checked(read(old/"control/protocol.json")["latents"])))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);forbid_optimizers()
    rt=Runtime(assets,index,manifest["initial_adapter"],manifest["audio_contract"]);file=weight(root,pub,"cf","C2")
    saved=torch.load(file,map_location="cuda",weights_only=True);rt.adapter.load_state_dict(saved["state"]);rt.adapter.eval();rt.adapter.requires_grad_(False);before=digest(rt.adapter)
    cfg=config["protected"];stored={};records=[];solves=[];native_records=[]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    try:
        for scope,pairs in (("train",cp["chosen_train_pairs"]),("val",manifest["validation_pairs"])):
            for case,pair in enumerate(pairs):
                uids=[pair["a"],pair["b"]]
                if any(index[u]["role"]!=scope for u in uids):raise ValueError("Population role changed")
                target,music,*_=rt.batch_self([uids[0]])
                noise=torch.randn(target.shape,device="cuda",generator=torch.Generator(device="cuda").manual_seed(config["seed"]+case+500))
                available=[];refs=[]
                for uid in uids:
                    cond,keys,plan,contract=rt.source(uid)
                    available.append(phase_reference(cond.value[0].numpy(),cond.known[0].numpy(),keys[0].numpy(),read(checked(read(old/"control/protocol.json")["phase_config"]))["phases"]))
                    refs.append(rt.self_data(uid)[2])
                common=available[0][1]&available[1][1];rt.state.update(enabled=False,capture=False)
                for tick in config["times"]:
                    t=torch.tensor([tick],device="cuda")
                    x=torch.from_numpy(latents["latent__"+str(case)+"__"+str(tick)]).cuda() if scope=="val" else rt.diffusion.q_sample(target,t,noise)
                    capture={};h=rt.model.cond_encoder.register_forward_hook(lambda module,args,value:capture.update(base=value.detach().clone()))
                    with sdpa_kernel(SDPBackend.MATH),torch.no_grad():base_pred=rt.diffusion.model_predictions(x,music,t,weight=2,clip_x_start=True)[1]
                    h.remove();zero=torch.zeros_like(capture["base"]);radius=control_norm(capture["base"],cfg["relative_rms"])
                    label=scope+"__"+str(case)+"__"+str(tick)
                    if not np.array_equal(base_pred.cpu().numpy(),old_arrays[label+"__base"]):raise ValueError("Frozen common state baseline changed")
                    def predict(z):
                        with injection(rt,"cond",z):return rt.diffusion.model_predictions(x,music,t,weight=2,clip_x_start=True)[1]
                    base=native(rt,base_pred);stored[label+"__base"]=base_pred.cpu().numpy()
                    raw=(base_pred-torch.as_tensor(rt.bias,device="cuda"))/torch.as_tensor(rt.scale,device="cuda")
                    base_rotation=_matrix(raw[...,7:].reshape(1,150,24,6)).transpose(-1,-2)[0].detach()
                    if scope=="val" and tick==19:
                        _,metrics=physical(rt,base_pred,music,out/(label+"__base"),assets);native_records.append(dict(case=case,arm="base",source=None,**metrics))
                    for s in (0,1):
                        variants={};trust={}
                        with sdpa_kernel(SDPBackend.MATH):
                            for kind in ("B1","B2"):
                                protected,seed,solve=guided(predict,zero,rt,base_rotation,available[s],radius,cfg,kind)
                                old_direction=old_arrays[label+"__"+str(s)+"__"+kind+"__direction"]
                                if not np.allclose(seed.cpu().numpy(),old_direction,rtol=1e-5,atol=1e-7):raise ValueError("Inference-only source gradient differs from frozen baseline")
                                arm="P"+kind;solve.update(scope=scope,case=case,timestep=tick,source=s,arm=arm)
                                solves.append(solve);variants[kind]=seed;variants[arm]=protected;trust[arm]=solve.get("numeric_trusted",False)
                                norm=protected.norm();support=torch.from_numpy(available[s][2].any(-1)[None,:,None]).cuda() if kind=="B2" else torch.ones((1,150,1),device="cuda",dtype=torch.bool)
                                variants["shrink_"+kind]=normalized_supported(seed,support,norm)
                                generator=torch.Generator(device="cuda").manual_seed(config["seed"]+case*1000+tick)
                                variants["random_"+arm]=normalized_supported(torch.randn(zero.shape,device="cuda",generator=generator),support,norm)
                            with torch.no_grad():
                                for arm,d in variants.items():
                                    pred=predict(d);motion=native(rt,pred)
                                    if arm in ("B1","B2") and not np.array_equal(pred.cpu().numpy(),old_arrays[label+"__"+str(s)+"__"+arm+"__predicted_x0"]):raise ValueError("Frozen B1/B2 prediction changed")
                                    own=source_error(base,refs[s],common)-source_error(motion,refs[s],common)
                                    wrong=source_error(base,refs[1-s],common)-source_error(motion,refs[1-s],common)
                                    record=dict(scope=scope,case=case,timestep=tick,source=s,arm=arm,own_gain_rad=own,wrong_gain_rad=wrong,directional_gain_rad=own-wrong,
                                        protection=protection(base,motion,config["protection"]),input_norm=float(d.double().norm()),budget_norm=radius,numeric_trusted=trust.get(arm,None))
                                    records.append(record);key=label+"__"+str(s)+"__"+arm;stored[key+"__direction"]=d.cpu().numpy();stored[key+"__predicted_x0"]=pred.cpu().numpy()
                                    if scope=="val" and tick==19 and arm in ("PB1","PB2"):
                                        _,metrics=physical(rt,pred,music,out/key,assets);native_records.append(dict(case=case,arm=arm,source=s,**metrics))
                    print("PROTECTED_CASE",scope,case,tick,flush=True)
        if digest(rt.adapter)!=before:raise ValueError("Frozen adapter changed")
    finally:rt.close()
    result=dict(records=records,solves=solves,qualification=qualify(records,solves,cfg),native_readouts=native_records,native_evaluations=len(native_records),
        native_passes=sum(r["native_ok"] for r in native_records),optimizer_calls=0,full_dances_generated=0,full_chain_started=False,mapping_training_started=False,operational_qualified=False)
    np.savez_compressed(out/"arrays.npz",**stored);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,frozen_control=ref(old/"control/protocol.json"),frozen_arrays=ref(old/"control/arrays.npz"),weights=ref(file),
        scope="rank-six Krylov projection, one linearization, actual clipped CFG2; condition512 only",matrix_free=True,
        mean_other_constraint="linear tangent mean Euclidean norm",global_feasibility_claimed=False,
        inference_reads="provided reference and current baseline prediction only; hidden target used only for evaluation"))
    if replay_from:
        previous=Path(replay_from)
        if read(previous/"result.json")!=result:raise ValueError("Protected readouts replay differs")
        same_arrays(previous/"arrays.npz",out/"arrays.npz");count=0
        for f in previous.glob("*__*/native.npy"):
            for name in ("normalized.npy","native.npy","native_fk.npy"):
                if not np.array_equal(np.load(f.parent/name),np.load(out/f.parent.name/name)):raise ValueError("Protected Native array differs")
                count+=1
            if read(f.parent/"physical.json")!=read(out/f.parent.name/"physical.json"):raise ValueError("Protected Native readout differs")
        write(out/"replay.json",dict(status="FRESH_PROTECTED_DIRECTION_REPLAY",optimizer_calls=0,solvers_rerun=len(solves),native_arrays_verified=count,exact_arrays=True))
    print("PROTECTED_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

"""Offline R7-A; paired identity references are not deployment negatives."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from dunhuang.generation.event_geometry import _matrix
from dunhuang.research.source_identifiability.g2 import Runtime,source_error
from dunhuang.research.source_capacity.capacity import native,injection
from dunhuang.research.source_capacity.numerics import normalized_supported
from dunhuang.research.protected_guidance.solver import basis,physical_coordinates
from dunhuang.research.reference_guidance.core import phase_reference,protection
from dunhuang.research.reference_guidance.control import control_norm
from dunhuang.research.source_expression.losses import reference_data,equalize
from .common import read,commit,fresh,prior,frozen,weight,same_arrays,forbid_optimizers
from .losses import errors,objective
from .solver import project
from .numerics import check
def projected(predict,zero,rt,base_rot,own,other,dynamic,norm,config,radius,protect_cfg,arm):
    z=zero.detach().clone().requires_grad_();pred=predict(z);e=errors(rt,pred,own,other)
    g_own=torch.autograd.grad(e[0],z,retain_graph=True)[0]
    g_rel=torch.autograd.grad(e[1],z,retain_graph=True)[0]
    g=torch.autograd.grad(objective(rt,pred,own,other,dynamic,norm,config,arm),z)[0]
    support=torch.ones((1,150,1),device=zero.device,dtype=torch.bool);seed=normalized_supported(-g,support,radius).detach()
    vectors,a,gram,target=basis(lambda v:physical_coordinates(rt,predict(v),base_rot),zero,seed,support,radius,protect_cfg["rank"])
    flat=torch.stack([v.flatten().double() for v in vectors])
    gains=torch.stack([-g_own.flatten().double(),-g_rel.flatten().double()])@flat.T*radius
    gains=gains.detach().cpu().numpy()
    coeff,meta=project(a,gram,target,gains,np.array([config["own_margin_rad"],config["relative_margin_rad"]]),protect_cfg)
    direction=sum((float(v)*b*radius for v,b in zip(coeff,vectors)),torch.zeros_like(zero)).detach()
    return direction,meta,np.r_[a@coeff,-(gains@coeff)/config["relative_margin_rad"]],dict(physical=a,gram=gram,target=target,gains=gains,coeff=coeff)
def run(config,preflight,out,replay_from=None):
    out=fresh(out);pre=read(Path(preflight)/"protocol.json")
    if pre["config"]!=config or not read(Path(preflight)/"result.json")["train_mechanism_admitted"]:raise ValueError("Preflight admission absent")
    r6root,r6,prep=prior(config);root,pub,p,manifest=frozen(r6);index={r["window_uid"]:r for r in manifest["rows"]}
    pairs=read(checked(pre["pairs"]));norm=read(checked(pre["normalizer"]));assets=read(checked(p["assets"]));old=dict(np.load(checked(pre["frozen_probe_arrays"])))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);forbid_optimizers()
    rt=Runtime(assets,index,manifest["initial_adapter"],manifest["audio_contract"]);file=weight(root,pub,"cf","C2")
    rt.adapter.load_state_dict(torch.load(file,map_location="cuda",weights_only=True)["state"]);rt.adapter.eval();rt.adapter.requires_grad_(False);before=digest(rt.adapter)
    stored={};records=[];solves=[];frozen_parities=0;cfg=r6["protected"]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    try:
        for cohort in config["cohorts"]:
            for case,pair in enumerate(pairs[cohort]):
                uids=[pair["a"],pair["b"]];target,music,*_=rt.batch_self([uids[0]])
                noise=torch.randn(target.shape,device="cuda",generator=torch.Generator(device="cuda").manual_seed(r6["seed"]+case+500))
                refs=[rt.self_data(uid)[2] for uid in uids];available=[]
                for uid in uids:
                    c,k,_,_=rt.source(uid);available.append(phase_reference(c.value[0].numpy(),c.known[0].numpy(),k[0].numpy(),read("configs/data_event_v3.json")["phases"]))
                mask=available[0][1]&available[1][1];rt.state.update(enabled=False,capture=False)
                for tick in config["times"]:
                    t=torch.tensor([tick],device="cuda");x=rt.diffusion.q_sample(target,t,noise);capture={}
                    h=rt.model.cond_encoder.register_forward_hook(lambda m,a,v:capture.update(base=v.detach().clone()))
                    with sdpa_kernel(SDPBackend.MATH),torch.no_grad():base_pred=rt.diffusion.model_predictions(x,music,t,weight=2,clip_x_start=True)[1]
                    h.remove();zero=torch.zeros_like(capture["base"]);radius=control_norm(capture["base"],cfg["relative_rms"])
                    label=cohort+"__"+str(case)+"__"+str(tick);assert np.array_equal(base_pred.cpu().numpy(),old[label+"__base"])
                    stored[label+"__base"]=base_pred.cpu().numpy();base=native(rt,base_pred)
                    raw=(base_pred-torch.as_tensor(rt.bias,device="cuda"))/torch.as_tensor(rt.scale,device="cuda")
                    base_rot=_matrix(raw[...,7:].reshape(1,150,24,6)).transpose(-1,-2)[0].detach()
                    def predict(z):
                        with injection(rt,"cond",z):return rt.diffusion.model_predictions(x,music,t,weight=2,clip_x_start=True)[1]
                    data=[reference_data(v,r6["selection_seed"]+case*100+tick) for v in available]
                    for source in (0,1):
                        own,other=data[source],data[1-source];shuffled=reference_data(available[source],r6["selection_seed"]+case*100+tick,True)
                        full={};meta={};expected={}
                        with sdpa_kernel(SDPBackend.MATH):
                            for arm,old_arm in (("A0","D0"),("A4","D4")):
                                key=label+"__"+str(source)+"__"+old_arm+"__full_budget"
                                full[arm]=torch.from_numpy(old[key+"__direction"]).cuda();meta[arm]=dict(status="FROZEN_R6_REFERENCE",new_solve=False)
                                with torch.no_grad():assert np.array_equal(predict(full[arm]).cpu().numpy(),old[key+"__predicted_x0"])
                                frozen_parities+=1
                            for arm in ("A1","A2","A3"):
                                d,m,expected[arm],small=projected(predict,zero,rt,base_rot,own,other,shuffled if arm=="A3" else own,norm,config,radius,cfg,arm)
                                full[arm]=d;meta[arm]=dict(m,new_solve=True)
                                for name,value in small.items():stored[label+"__"+str(source)+"__"+arm+"__small_"+name]=value
                            matched,dose=equalize(full)
                            def f(z):
                                value=predict(z)
                                return torch.cat([physical_coordinates(rt,value,base_rot),errors(rt,value,own,other)/config["relative_margin_rad"]])
                            for arm in config["arms"]:
                                d=matched[arm];jp,numeric=check(f,zero,d,cfg)
                                linear_error=None
                                if jp is not None and arm in expected:
                                    ratio=float(d.double().norm())/max(float(full[arm].double().norm()),1e-12)
                                    wanted=torch.from_numpy(expected[arm]*ratio).to(jp)
                                    linear_error=float((jp-wanted).norm()/jp.norm().clamp_min(1e-12))
                                    numeric["numeric_trusted"]&=linear_error<1e-4
                                solve=dict(meta[arm],cohort=cohort,case=case,timestep=tick,source=source,arm=arm,numerics=numeric,linear_combination_relative_error=linear_error)
                                solves.append(solve)
                                with torch.no_grad():
                                    for dose_kind,delta in (("full_budget",full[arm]),("common_norm",d)):
                                        pred=predict(delta);motion=native(rt,pred);a=source_error(base,refs[source],mask)-source_error(motion,refs[source],mask);b=source_error(base,refs[1-source],mask)-source_error(motion,refs[1-source],mask)
                                        records.append(dict(cohort=cohort,case=case,timestep=tick,source=source,arm=arm,dose_kind=dose_kind,own_gain_rad=a,other_gain_rad=b,directional_gain_rad=a-b,
                                            protection=protection(base,motion,r6["protection"]),input_norm=float(delta.double().norm()),budget_norm=radius,common_norm=dose,
                                            response_status="NOT_EVALUABLE_ZERO_COMMON_DOSE" if dose_kind=="common_norm" and dose==0 else "MEASURED",
                                            numeric_trusted=numeric["numeric_trusted"] if dose_kind=="common_norm" else None))
                                        key=label+"__"+str(source)+"__"+arm+"__"+dose_kind
                                        stored[key+"__direction"]=delta.cpu().numpy();stored[key+"__predicted_x0"]=pred.cpu().numpy()
                    print("R7_CASE",cohort,case,tick,flush=True)
        if digest(rt.adapter)!=before:raise ValueError("Frozen adapter changed")
    finally:rt.close()
    counts={}
    for cohort in config["cohorts"]:
        counts[cohort]={}
        for arm in config["arms"]:
            rs=[r for r in records if r["cohort"]==cohort and r["arm"]==arm and r["dose_kind"]=="common_norm"]
            nonzero=[r for r in rs if r["common_norm"]>0]
            joint=lambda r:r["own_gain_rad"]>config["own_margin_rad"] and r["directional_gain_rad"]>config["relative_margin_rad"] and r["protection"]["protected"] and r["numeric_trusted"]
            counts[cohort][arm]=dict(registered_denominator=len(rs),joint_trusted=sum(joint(r) for r in rs),zero_common_dose=len(rs)-len(nonzero),
                nonzero_conditional_diagnostic_denominator=len(nonzero),nonzero_joint=sum(joint(r) for r in nonzero),protected=sum(r["protection"]["protected"] for r in rs),
                common_mode=sum(r["own_gain_rad"]>config["own_margin_rad"] and r["directional_gain_rad"]<=config["relative_margin_rad"] and r["other_gain_rad"]>0 for r in rs))
    result=dict(records=records,solves=solves,qualification=counts,new_projection_solves=sum(s["new_solve"] for s in solves),frozen_prediction_parities=frozen_parities,
        independent_evaluation="NOT_EVALUABLE_NO_REGISTERED_UNEXPOSED_POPULATION",operational_qualified=False,optimizer_updates=0,native_calls=0,full_generation_calls=0,new_mapping_training_started=False)
    np.savez_compressed(out/"arrays.npz",**stored);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,preflight=ref(Path(preflight)/"protocol.json"),weights=ref(file),input_scope="two provided visible sources, offline identity only",sampling="same frozen R6 q states, no Val/Test objective probes"))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result:raise ValueError("R7 readout replay differs")
        same_arrays(prev/"arrays.npz",out/"arrays.npz");write(out/"replay.json",dict(status="FRESH_R7_PROJECTION_REPLAY",optimizer_calls=0,new_projection_solves_rerun=result["new_projection_solves"],exact_arrays=True))
    print("R7_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--preflight",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.preflight,a.out,a.replay_from)

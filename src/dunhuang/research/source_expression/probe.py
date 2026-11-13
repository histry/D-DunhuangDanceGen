"""R6-P1: fixed expression targets in the same protected condition domain."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from dunhuang.generation.event_geometry import _matrix
from dunhuang.research.source_capacity.capacity import native,injection
from dunhuang.research.source_capacity.numerics import normalized_supported,check_derivatives,trusted
from dunhuang.research.protected_guidance.solver import basis,project,physical_coordinates
from dunhuang.research.reference_guidance.control import control_norm
from dunhuang.research.reference_guidance.core import phase_reference,protection
from dunhuang.research.source_identifiability.g2 import Runtime,source_error
from .common import read,commit,fresh,prior,frozen,weight,same_arrays,forbid_optimizers
from .losses import reference_data,expression_loss,equalize,components

def projected(predict,zero,rt,base_rotation,loss,radius,cfg,seed=None):
    support=torch.ones((1,150,1),device="cuda",dtype=torch.bool)
    if seed is None:
        z=zero.detach().clone().requires_grad_()
        g=torch.autograd.grad(loss(predict(z)),z)[0];seed=normalized_supported(-g,support,radius).detach()
    vectors,a,gram,target=basis(lambda z:physical_coordinates(rt,predict(z),base_rotation),zero,seed,support,radius,cfg["rank"])
    coef,solve=project(a,gram,target,cfg)
    d=sum((float(v)*b*radius for v,b in zip(coef,vectors)),torch.zeros_like(zero)).detach()
    return d,solve,a@coef

def run(config,prepared,diagnosis,out,replay_from=None):
    out=fresh(out);prep=read(Path(prepared)/"protocol.json");diag=read(Path(diagnosis)/"result.json")
    if prep["config"]!=config or not diag["gate"]["train_only_expression_probe_admitted"]:raise ValueError("R6 preconditions incomplete")
    root,pub,p,manifest=frozen(config);old,oldpub=prior(config);index={r["window_uid"]:r for r in manifest["rows"]}
    pairs=read(checked(prep["pairs"]));norm=read(checked(prep["normalizer"]));assets=read(checked(p["assets"]));old_arrays=dict(np.load(old/"protected/arrays.npz"))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);forbid_optimizers()
    rt=Runtime(assets,index,manifest["initial_adapter"],manifest["audio_contract"]);file=weight(root,pub,"cf","C2")
    rt.adapter.load_state_dict(torch.load(file,map_location="cuda",weights_only=True)["state"]);rt.adapter.eval();rt.adapter.requires_grad_(False);before=digest(rt.adapter)
    records=[];solves=[];stored={};cfg=config["protected"];arms=config["probe"]["arms"]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    try:
        for cohort,pop in pairs.items():
            for case,pair in enumerate(pop):
                uids=[pair["a"],pair["b"]];target,music,*_=rt.batch_self([uids[0]])
                noise=torch.randn(target.shape,device="cuda",generator=torch.Generator(device="cuda").manual_seed(config["seed"]+case+500))
                refs=[rt.self_data(uid)[2] for uid in uids];available=[]
                for uid in uids:
                    c,k,plan,contract=rt.source(uid);available.append(phase_reference(c.value[0].numpy(),c.known[0].numpy(),k[0].numpy(),read("configs/data_event_v3.json")["phases"]))
                common=available[0][1]&available[1][1];rt.state.update(enabled=False,capture=False)
                for tick in config["times"]:
                    t=torch.tensor([tick],device="cuda");x=rt.diffusion.q_sample(target,t,noise)
                    capture={};h=rt.model.cond_encoder.register_forward_hook(lambda m,a,v:capture.update(base=v.detach().clone()))
                    with sdpa_kernel(SDPBackend.MATH),torch.no_grad():base_pred=rt.diffusion.model_predictions(x,music,t,weight=2,clip_x_start=True)[1]
                    h.remove();zero=torch.zeros_like(capture["base"]);radius=control_norm(capture["base"],cfg["relative_rms"])
                    label=cohort+"__"+str(case)+"__"+str(tick);stored[label+"__base"]=base_pred.cpu().numpy();base=native(rt,base_pred)
                    if cohort=="calibration" and not np.array_equal(base_pred.cpu().numpy(),old_arrays["train__"+str(case)+"__"+str(tick)+"__base"]):raise ValueError("Frozen Train base state changed")
                    raw=(base_pred-torch.as_tensor(rt.bias,device="cuda"))/torch.as_tensor(rt.scale,device="cuda");base_rot=_matrix(raw[...,7:].reshape(1,150,24,6)).transpose(-1,-2)[0].detach()
                    def predict(z):
                        with injection(rt,"cond",z):return rt.diffusion.model_predictions(x,music,t,weight=2,clip_x_start=True)[1]
                    for source in (0,1):
                        data=reference_data(available[source],config["selection_seed"]+case*100+tick)
                        shuffled=reference_data(available[source],config["selection_seed"]+case*100+tick,True)
                        full={};meta={};linear={}
                        with sdpa_kernel(SDPBackend.MATH):
                            for arm in arms:
                                current=shuffled if arm=="D3" else data
                                def loss(pred):return expression_loss(rt,pred,current,norm,config["probe"],arm)
                                seed=None
                                if arm=="D4":
                                    generator=torch.Generator(device="cuda").manual_seed(config["selection_seed"]+case*1000+tick)
                                    seed=normalized_supported(torch.randn(zero.shape,device="cuda",generator=generator),torch.ones((1,150,1),device="cuda",dtype=torch.bool),radius)
                                d,solve,expected=projected(predict,zero,rt,base_rot,loss,radius,cfg,seed)
                                if cohort=="calibration" and arm=="D0":
                                    previous=old_arrays["train__"+str(case)+"__"+str(tick)+"__"+str(source)+"__PB1__direction"]
                                    if not np.allclose(d.cpu().numpy(),previous,rtol=1e-4,atol=1e-6):raise ValueError("D0 differs from frozen PB1")
                                full[arm]=d;meta[arm]=solve;linear[arm]=expected
                            matched,dose=equalize(full)
                            for arm in arms:
                                d=matched[arm]
                                def f(z):return physical_coordinates(rt,predict(z),base_rot)
                                _,reference_jvp=torch.autograd.functional.jvp(f,zero,d)
                                readout=reference_jvp.detach()/reference_jvp.norm().clamp_min(1e-12)
                                jp,g,derivative=check_derivatives(f,zero,d,readout,cfg["fd_steps"])
                                expected=torch.from_numpy(linear[arm]*(float(d.double().norm())/max(float(full[arm].double().norm()),1e-12))).to(jp)
                                error=float((jp-expected).norm()/jp.norm().clamp_min(1e-12))
                                solve=dict(meta[arm],derivatives=derivative,numeric_trusted=trusted(derivative,cfg) and error<1e-4 and meta[arm]["status"]=="SOLVED_FINITE_SUBSPACE",
                                    linear_combination_relative_error=error,cohort=cohort,case=case,timestep=tick,source=source,arm=arm)
                                solves.append(solve)
                                with torch.no_grad():
                                    for dose_kind,delta in (("full_budget",full[arm]),("common_norm",d)):
                                        pred=predict(delta);motion=native(rt,pred);own=source_error(base,refs[source],common)-source_error(motion,refs[source],common);wrong=source_error(base,refs[1-source],common)-source_error(motion,refs[1-source],common)
                                        expr=components(rt,pred,data);base_expr=components(rt,base_pred,data)
                                        record=dict(cohort=cohort,case=case,timestep=tick,source=source,arm=arm,dose_kind=dose_kind,own_gain_rad=own,wrong_gain_rad=wrong,directional_gain_rad=own-wrong,
                                            velocity_gain_squared_radps=float(base_expr["velocity"]-expr["velocity"]),stage_pose_gain_squared_rad=float(base_expr["event"]-expr["event"]),
                                            protection=protection(base,motion,config["protection"]),input_norm=float(delta.double().norm()),budget_norm=radius,common_norm=dose,numeric_trusted=solve["numeric_trusted"] if dose_kind=="common_norm" else None)
                                        records.append(record);key=label+"__"+str(source)+"__"+arm+"__"+dose_kind
                                        stored[key+"__direction"]=delta.cpu().numpy();stored[key+"__predicted_x0"]=pred.cpu().numpy()
                    print("R6_EXPRESSION_CASE",cohort,case,tick,flush=True)
        if digest(rt.adapter)!=before:raise ValueError("Frozen adapter changed")
    finally:rt.close()
    qualification={}
    for cohort in pairs:
        counts={}
        for arm in arms:
            rs=[r for r in records if r["cohort"]==cohort and r["arm"]==arm and r["dose_kind"]=="common_norm"]
            counts[arm]=dict(denominator=len(rs),joint_trusted=sum(r["own_gain_rad"]>.001 and r["directional_gain_rad"]>.001 and r["protection"]["protected"] and r["numeric_trusted"] for r in rs),
                protected=sum(r["protection"]["protected"] for r in rs),directional=sum(r["directional_gain_rad"]>.001 for r in rs))
        qualification[cohort]=counts
    result=dict(records=records,solves=solves,qualification=qualification,normalizer_fit_calls=0,optimizer_calls=0,native_calls=0,full_generation_calls=0,
        independent_model_generalization="NOT_QUALIFIED_TRAIN_EXPOSURE",full_chain_started=False,new_mapping_training_started=False,operational_qualified=False)
    np.savez_compressed(out/"arrays.npz",**stored);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,prepared=ref(Path(prepared)/"protocol.json"),diagnosis=ref(Path(diagnosis)/"protocol.json"),weights=ref(file),
        loss_weights="unit-weight components with old-Train reference-only scales",common_dose="minimum of five projected norms; no expansion",
        population="four fixed old Train pairs and four disjoint known-Train probe pairs; not a fresh blind source"))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result:raise ValueError("Expression readouts differ")
        same_arrays(prev/"arrays.npz",out/"arrays.npz");write(out/"replay.json",dict(status="FRESH_EXPRESSION_SOLVE_REPLAY",optimizer_calls=0,solves_rerun=len(solves),exact_arrays=True))
    print("R6_EXPRESSION_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--prepared",required=True);p.add_argument("--diagnosis",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.prepared,a.diagnosis,a.out,a.replay_from)

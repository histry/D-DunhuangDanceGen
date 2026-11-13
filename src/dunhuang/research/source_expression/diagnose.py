"""R6-P0: frozen responsibility matrix and fixed-active-set clip diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from dunhuang.generation.event_geometry import _matrix
from dunhuang.research.source_identifiability.g2 import Runtime
from dunhuang.research.source_capacity.capacity import injection
from dunhuang.research.source_capacity.numerics import check_derivatives,trusted
from dunhuang.research.protected_guidance.solver import physical_coordinates
from .common import read,commit,fresh,prior,frozen,weight,same_arrays,forbid_optimizers

def responsibility(record,solver=None):
    tags=[]
    if record["own_gain_rad"]<=.001:tags.append("INSUFFICIENT_OWN_POSE_GAIN")
    if record["directional_gain_rad"]<=.001:
        tags.append("COMMON_MODE_GAIN" if record["wrong_gain_rad"]>0 and record["own_gain_rad"]>0 else "SOURCE_DIRECTION_UNRESOLVED")
    if not record["protection"]["protected"]:tags.append("PROTECTION_CONFLICT")
    if solver and not solver.get("numeric_trusted",False):tags.append("NUMERIC_UNKNOWN")
    return tags or ["MEASURED_PASS"]

def clip_mask(raw):
    # torch.clamp has derivative1 at exact +/-1.
    return (raw>=-1)&(raw<=1)

def run(config,out,replay_from=None):
    out=fresh(out);root,pub=prior(config);old=read(root/"protected/result.json");arrays=dict(np.load(root/"protected/arrays.npz"))
    solves={(r["scope"],r["case"],r["timestep"],r["source"],r["arm"]):r for r in old["solves"]}
    matrix=[]
    for r in old["records"]:
        if r["arm"] not in ("PB1","PB2"):continue
        base=next(v for v in old["records"] if (v["scope"],v["case"],v["timestep"],v["source"],v["arm"])==(r["scope"],r["case"],r["timestep"],r["source"],"B"+r["arm"][-1]))
        key=r["scope"]+"__"+str(r["case"])+"__"+str(r["timestep"])+"__"+str(r["source"])+"__"
        a=arrays[key+"B"+r["arm"][-1]+"__direction"].astype(float);b=arrays[key+r["arm"]+"__direction"].astype(float)
        cosine=float((a*b).sum()/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
        limits=config["protection"]
        matrix.append(dict(scope=r["scope"],case=r["case"],timestep=r["timestep"],source=r["source"],arm=r["arm"],
            responsibility=responsibility(r,solves[(r["scope"],r["case"],r["timestep"],r["source"],r["arm"])]),
            own_gain_rad=r["own_gain_rad"],wrong_gain_rad=r["wrong_gain_rad"],directional_gain_rad=r["directional_gain_rad"],
            source_common_mode_fraction=r["wrong_gain_rad"]/max(abs(r["own_gain_rad"]),1e-12),direction_cosine_to_raw=cosine,
            relative_norm=float(np.linalg.norm(b)/max(np.linalg.norm(a),1e-12)),directional_gain_change=r["directional_gain_rad"]-base["directional_gain_rad"],
            protection_margins={n:limits[n]-r["protection"][n] for n in ("root_rms_m","contact_max","non_target_mean_rad")},
            amplitude_insufficiency="UNKNOWN_NO_NEW_DOSE_SWEEP"))
    _,source_pub,p,manifest=frozen(config);assets=read(checked(p["assets"]));index={r["window_uid"]:r for r in manifest["rows"]}
    rt=Runtime(assets,index,manifest["initial_adapter"],manifest["audio_contract"]);file=weight(Path(config["source"]),source_pub,"cf","C2")
    saved=torch.load(file,map_location="cuda",weights_only=True);rt.adapter.load_state_dict(saved["state"]);rt.adapter.eval();rt.adapter.requires_grad_(False);before=digest(rt.adapter)
    reference=read(root/"protected/protocol.json");cp=read(checked(read(checked(reference["frozen_control"]))["capacity_protocol"]))
    latents=dict(np.load(checked(read(checked(reference["frozen_control"]))["latents"])))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);forbid_optimizers()
    records=[];stored={};cfg=config["protected"]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    try:
        pair=manifest["validation_pairs"][0];_,music,*_=rt.batch_self([pair["a"]]);rt.state.update(enabled=False,capture=False)
        for tick in (499,19):
            x=torch.from_numpy(latents["latent__0__"+str(tick)]).cuda();t=torch.tensor([tick],device="cuda")
            capture={};h=rt.model.cond_encoder.register_forward_hook(lambda m,a,v:capture.update(base=v.detach().clone()))
            with sdpa_kernel(SDPBackend.MATH),torch.no_grad():base_raw=rt.model.guided_forward(x,music,t,2)
            h.remove();base=base_raw.clamp(-1,1);zero=torch.zeros_like(capture["base"])
            if not np.array_equal(base.cpu().numpy(),arrays["val__0__"+str(tick)+"__base"]):raise ValueError("Frozen baseline changed")
            raw_motion=(base-torch.as_tensor(rt.bias,device="cuda"))/torch.as_tensor(rt.scale,device="cuda")
            rot=_matrix(raw_motion[...,7:].reshape(1,150,24,6)).transpose(-1,-2)[0].detach();active=clip_mask(base_raw).detach()
            def raw(z):
                with injection(rt,"cond",z):return rt.model.guided_forward(x,music,t,2)
            with sdpa_kernel(SDPBackend.MATH):
                for source in (0,1):
                    for arm in ("PB1","PB2"):
                        key="val__0__"+str(tick)+"__"+str(source)+"__"+arm
                        d=torch.from_numpy(arrays[key+"__direction"]).cuda()
                        def clipped(z):return physical_coordinates(rt,raw(z).clamp(-1,1),rot)
                        def fixed(z):return physical_coordinates(rt,base+active*(raw(z)-base_raw),rot)
                        _,seed_jvp=torch.autograd.functional.jvp(clipped,zero,d);target=seed_jvp.detach()/seed_jvp.norm().clamp_min(1e-12)
                        j,g,real=check_derivatives(clipped,zero,d,target,cfg["fd_steps"])
                        jf,gf,frozen_clip=check_derivatives(fixed,zero,d,target,cfg["fd_steps"])
                        transitions=[]
                        with torch.no_grad():
                            for step in cfg["fd_steps"]:
                                vals=[raw(sign*step*d) for sign in (-1,1)]
                                changed=[int((clip_mask(v)!=active).sum()) for v in vals]
                                ulp=torch.nextafter(base_raw,torch.full_like(base_raw,float("inf")))-base_raw
                                transitions.append(dict(h=step,active_set_changes_minus=changed[0],active_set_changes_plus=changed[1],
                                    raw_response_rms=float((vals[1]-vals[0]).square().mean().sqrt()),
                                    raw_fp32_ulp_rms=float(ulp.square().mean().sqrt()),
                                    changed_fraction=float(sum(changed)/(2*active.numel()))))
                        real_pass=trusted(real,cfg);fixed_pass=trusted(frozen_clip,cfg)
                        records.append(dict(case=0,timestep=tick,source=source,arm=arm,actual_clipped=real,fixed_active_diagnostic=frozen_clip,
                            actual_numeric_trusted=real_pass,fixed_active_numeric_trusted=fixed_pass,
                            jvp_identity_relative_error=float((j-jf).norm()/j.norm().clamp_min(1e-12)),
                            baseline_clipped_components=int((~active).sum()),baseline_components=active.numel(),
                            transitions=transitions,attribution="ACTIVE_SET_SWITCHING_SUPPORTED" if not real_pass and fixed_pass else "NO_UNIQUE_NUMERIC_CAUSE",
                            qualification_override=False))
                        stored[key+"__jvp"]=j.cpu().numpy();stored[key+"__fixed_jvp"]=jf.cpu().numpy();stored[key+"__raw"]=base_raw.cpu().numpy()
                        print("R6_CLIP_DIAGNOSTIC",tick,source,arm,flush=True)
        if digest(rt.adapter)!=before:raise ValueError("Frozen adapter changed")
    finally:rt.close()
    result=dict(responsibility_matrix=matrix,numeric_diagnostics=records,old_numeric_trusted=sum(r.get("numeric_trusted",False) for r in old["solves"]),
        old_numeric_denominator=len(old["solves"]),gate=dict(frozen_inputs_verified=True,train_only_expression_probe_admitted=True,original_numeric_qualification_unchanged=True),
        optimizer_calls=0,full_generation_calls=0,operational_qualified=False)
    np.savez_compressed(out/"arrays.npz",**stored);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,frozen_result=ref(root/"protected/result.json"),frozen_arrays=ref(root/"protected/arrays.npz"),weights=ref(file),
        diagnostics_only="frozen active-set surrogate never enters sampling or qualification",fixed_cases="Val case0, times499/19, both sources, PB1/PB2"))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result:raise ValueError("Responsibility/numeric replay differs")
        same_arrays(prev/"arrays.npz",out/"arrays.npz");write(out/"replay.json",dict(status="FRESH_FROZEN_FAILURE_REPLAY",optimizer_calls=0,exact_arrays=True,cases=8))
    print("R6_DIAGNOSIS_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

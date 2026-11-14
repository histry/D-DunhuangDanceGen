"""Frozen input-space reference guidance, only after the registered P1 gate."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.contracts import MotionAdapter,sixd
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.masked_denoiser.common import setup,make,checkpoint_load
from dunhuang.research.masked_denoiser.rollout import sample,replay_arrays
from dunhuang.research.boundary_context.common import masks
from dunhuang.research.motion_bridge.model import rotations
from .common import population
from .states import extract,design,reference
from .readout import donors

def objective(pred,ref,scale,bias):
    p=(pred-bias)/scale;r=(ref-bias)/scale
    root=((p[:,60:90,4:7]-r[:,60:90,4:7])/.1).square().mean()
    P=rotations(p[:,60:90,7:13]);R=rotations(r[:,60:90,7:13])
    return root+(P-R).square().sum((-1,-2)).mean()/.3**2

def run(cfg,root,out,replay=None):
    out=fresh(out);forbid_optimizers();root=Path(root);base=read(cfg["base_config"])
    heads=read(root/"readout/result.json");assert heads["selected_family"]=="A2"
    assert heads["qualification"]["A2"]["qualified"]
    _,_,hold,_,_=population(base)
    groups=sorted(set(r["source_group"] for r in hold));rows=[]
    for g in groups:
        rr=[r for r in hold if r["source_group"]==g]
        rows.append(min(rr,key=lambda r:hashlib.sha256((str(cfg["seed"])+r["raw_motion"]["sha256"]+str(r["start_frame"])).encode()).hexdigest()))
    assert len(rows)==cfg["cases"]==8
    oldcfg=read(base["r13_config"]);oldcfg=dict(oldcfg,self_cache_steps=[24,49])
    # Model checkpoint schema is the original R13 config; sampler probes are a separate setting.
    assets,_,_,_,_,data,model,diffusion=setup(read(base["r13_config"]))
    wrapper=make(read(base["r13_config"]),model)
    checkpoint=read(Path(base["r13_root"])/"main/result.json")["checkpoint"]
    checkpoint_load(read(base["r13_config"]),wrapper,checkpoint,assets);before=wrapper.digest()
    for param in wrapper.parameters():param.requires_grad_(False)
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    models={f:dict(np.load(checked(heads["models"][f]))) for f in ("A0","A2")}
    motions=[data.native(r) for r in rows];features=[extract(x) for x in motions]
    donor=donors(rows,cfg["seed"]);refs={};bindings=[];derivatives=[]
    for i,row in enumerate(rows):
        y,music=data.batch([row]);known,_=masks(y,60,90)
        folder=out/("case_"+str(i))
        sample(diffusion,wrapper,y,music,y,cfg["seed"]+i,oldcfg,folder,counterfactuals=False)
        if replay:replay_arrays(folder,Path(replay)/folder.name)
        z=np.load(folder/"trace.npz");saved={}
        for arm in cfg["reference_arms"]:
            f="A0" if arm=="raw_A0" else "A2";feature=features[int(donor[i])] if arm=="shuffled_A2" else features[i]
            # Shuffle only explicit state. Raw observations and true hard anchors remain fixed.
            raw,mask,_,_=features[i];_,_,state,valid=feature
            xx=design(raw,mask,state,valid,f);m=models[f]
            pp=((xx-m["mean"])/m["std"])@m["coef"]+m["target_mean"]
            p,R=reference(motions[i],pp);r=motions[i].copy();r[60:90,4:7]=p;r[60:90,7:13]=sixd(R)
            origin=r[0,4:7].copy();origin[1]=0
            encoded=MotionAdapter.to_edge(r,origin)*data.scale+data.bias
            # Only the unknown Root/Root rotation coordinates enter the reference objective.
            refs[str(i)+"_"+arm]=np.c_[p,sixd(R)]
            rr=torch.tensor(encoded[None],device="cuda",dtype=torch.float32)
            for timestep in cfg["timesteps"]:
                j=int(np.flatnonzero(z["t"]==timestep)[0])
                x=torch.tensor(z["xt"][j:j+1],device="cuda",requires_grad=True);t=torch.tensor([timestep],device="cuda",dtype=torch.long)
                pred=wrapper(x,music,t,y,known);loss=objective(pred,rr,scale,bias)
                grad,=torch.autograd.grad(loss,x);active=torch.zeros_like(grad);active[:,60:90,4:13]=1
                grad=grad*active;rms=grad.square().sum().div(270).sqrt()
                direction=-grad/rms.clamp(min=1e-20)
                with torch.no_grad():
                    delta=direction*cfg["input_root_rms"]
                    guided=wrapper(x+delta,music,t,y,known)
                    eps=cfg["derivative_epsilon"]
                    hi=objective(wrapper(x+direction*eps,music,t,y,known),rr,scale,bias)
                    lo=objective(wrapper(x-direction*eps,music,t,y,known),rr,scale,bias)
                    actual=(hi-lo)/(2*eps);analytic=(grad*direction).sum()
                    relative=float(abs(actual-analytic)/torch.maximum(abs(actual),abs(analytic)).clamp(min=1e-6))
                key=arm+"_"+str(timestep);saved[key]=guided[0].cpu().numpy()
                saved["baseline_"+str(timestep)]=pred[0].detach().cpu().numpy()
                derivatives.append(dict(case=i,arm=arm,t=timestep,objective_before=float(loss.detach()),
                    objective_after=float(objective(guided,rr,scale,bias)),gradient_rms=float(rms),
                    derivative_relative_error=relative,derivative_analytic=float(analytic),derivative_numeric=float(actual)))
        np.savez_compressed(folder/"probes.npz",**saved)
        if replay:
            from dunhuang.research.source_tasks.common import same_arrays
            same_arrays(folder/"probes.npz",Path(replay)/folder.name/"probes.npz")
        bindings.append(dict(case=i,row=row,probe=str(folder/"probes.npz"),trace=str(folder/"trace.npz")))
        print("R14_CAPACITY",i,flush=True)
    np.savez_compressed(out/"references.npz",**refs)
    assert wrapper.digest()==before
    result=dict(bindings=bindings,derivatives=derivatives,optimizer_steps=0,model_unchanged=True,
        baseline_full_window_generations=8,guided_full_window_generations=0,neural_parameter_updates=0,
        scope="single frozen baseline-policy state probes; not guided full-chain generations",operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        head_models=heads["models"],rows=rows,donors=donor.tolist(),r13_checkpoint=checkpoint,
        labels="paired hidden motion used only by sampler target container and later scoring; reference uses observed sides only"))
    if replay:
        old=read(Path(replay)/"result.json")
        assert result["derivatives"]==old["derivatives"]
        from dunhuang.research.source_tasks.common import same_arrays
        same_arrays(out/"references.npz",Path(replay)/"references.npz")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True)
    p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

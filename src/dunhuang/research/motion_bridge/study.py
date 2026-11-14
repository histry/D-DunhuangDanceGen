"""Registered Train-only capacity and actual-inpainting cache, before DEV reads."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import Data,load_model,digest,masks,fixed_rows
from dunhuang.sampling import constrained_ddim
from .model import Injection,observation
from .objective import capacity_objective

def population(cfg):
    pf=Path(cfg["preflight"])
    assert read(pf/"result.json")["bridge_training_data_ready"]
    rows=read(pf/"rows.json");train=[r for r in rows if r["role"]=="train"]
    groups=sorted(set(r["source_group"] for r in train))
    import hashlib
    groups.sort(key=lambda s:hashlib.sha256((str(cfg["seed"])+s).encode()).hexdigest())
    assert len(train)==1833 and len(groups)==48
    parts={"capacity_fit":groups[:4],"capacity_holdout":groups[4:8],"monitor":groups[8:12],"fit":groups[12:]}
    assert not set(groups)&set(r["source_group"] for r in rows if r["role"]!="train")
    def select(name,n):
        return fixed_rows([r for r in train if r["source_group"] in parts[name]],"train",n,cfg["data_seed"])
    return read(checked(read(pf/"protocol.json")["assets"])),train,parts,select

def capacity(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers()
    assets,train,groups,select=population(cfg)
    rows=select("capacity_fit",4)+select("capacity_holdout",4)
    model,diffusion,_=load_model(assets);before=digest(model)
    data=Data(np.load(checked(assets["scaler"])))
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    targets,music=data.batch(rows);known,_=masks(targets,60,90)
    wrong=targets.roll(1,0)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,assets=ref(checked(read(Path(cfg["preflight"])/"protocol.json")["assets"])),
        preflight=ref(Path(cfg["preflight"])/"protocol.json"),rows=ref(Path(cfg["preflight"])/"rows.json"),groups=groups,population=rows,
        perturbation="one normalized feature-space negative-gradient step; matched full 150x512 RMS; NOT model training",
        direction_target="observed 8+8 Root/rotations only",val_tensor_reads=0))
    records=[];arrays={}
    for i in range(8):
        y=targets[i:i+1];m=music[i:i+1];k=known[i:i+1];w=wrong[i:i+1]
        for timestep in cfg["capacity_timesteps"]:
            torch.manual_seed(cfg["seed"]+i*1000+timestep);torch.cuda.manual_seed_all(cfg["seed"]+i*1000+timestep)
            t=torch.tensor([timestep],device="cuda");xt=diffusion.q_sample(y,t)
            for site in cfg["sites"]:
                hook=Injection(model,site);z=torch.zeros(1,150,512,device="cuda",requires_grad=True)
                base=hook.predict(xt,m,t,z)
                with torch.no_grad():
                    reference=diffusion.model_predictions(xt,m,t,weight=2,clip_x_start=True)[1]
                    assert torch.equal(base.detach(),reference)
                initial=float(capacity_objective(base,y,k,scale,bias))
                derivatives=[]
                for truth in (y,w):
                    d=torch.zeros_like(z,requires_grad=True);pred=hook.predict(xt,m,t,d)
                    obj=capacity_objective(pred,truth,k,scale,bias)
                    grad=torch.autograd.grad(obj,d)[0]
                    assert torch.isfinite(grad).all()
                    derivatives.append(grad.detach())
                random=torch.randn(z.shape,device="cuda")
                for control,direction in (("correct",-derivatives[0]),("wrong",-derivatives[1]),("random",random)):
                    rms=direction.square().mean().sqrt()
                    perturb=direction/rms.clamp(min=1e-12)*cfg["perturbation_rms"]
                    with torch.no_grad():
                        pred=hook.predict(xt,m,t,perturb)
                        value=float(capacity_objective(pred,y,k,scale,bias))
                    row=dict(case=i,role="capacity_fit" if i<4 else "capacity_holdout",site=site,t=timestep,control=control,
                        before=initial,after=value,relative_gain=(initial-value)/max(initial,1e-12),
                        perturbation_rms=float(perturb.square().mean().sqrt()),gradient_norm=float(derivatives[0].norm()),
                        unknown_mse=float((pred[:,60:90,4:]-y[:,60:90,4:]).square().mean()))
                    records.append(row);arrays[str(len(records))]=pred.cpu().numpy()
                hook.close()
        print("R12_CAPACITY",i,flush=True)
    scores={}
    for site in cfg["sites"]:
        summaries=[]
        for i in range(4,8):
            means={c:float(np.mean([r["after"] for r in records if r["site"]==site and r["case"]==i and r["control"]==c])) for c in ("correct","wrong","random")}
            initial=float(np.mean([r["before"] for r in records if r["site"]==site and r["case"]==i and r["control"]=="correct"]))
            summaries.append(dict(case=i,**means,before=initial))
        positive=sum(r["correct"]<r["before"] for r in summaries)
        wins=sum(r["correct"]<r["wrong"] and r["correct"]<r["random"] for r in summaries)
        scores[site]=dict(cases=summaries,positive_cases=positive,correct_control_wins=wins,
            qualified=positive>=cfg["capacity_gate"]["minimum_positive_cases"] and wins>=cfg["capacity_gate"]["minimum_correct_vs_wrong_cases"])
    selected=next((s for s in cfg["site_preference"] if scores[s]["qualified"]),None)
    assert digest(model)==before
    result=dict(records=records,scores=scores,selected_site=selected,training_admitted=selected is not None,
        optimizer_steps=0,backbone_digest=before,backbone_unchanged=True,val_tensor_reads=0,operational_qualified=False)
    np.savez_compressed(out/"predictions.npz",**arrays);write(out/"result.json",result)
    if replay:
        assert result==read(Path(replay)/"result.json")
        from dunhuang.research.source_tasks.common import same_arrays
        same_arrays(out/"predictions.npz",Path(replay)/"predictions.npz")
    print("R12_CAPACITY_DONE",selected,scores,flush=True)

def cache(cfg,capacity_root,out,replay=None):
    out=fresh(out);forbid_optimizers()
    assert read(Path(capacity_root)/"result.json")["training_admitted"]
    assets,train,groups,select=population(cfg);data=Data(np.load(checked(assets["scaler"])))
    fit=[r for r in train if r["source_group"] in groups["fit"]]
    # Statistics use FIT groups only, never capacity, monitor or DEV tensors.
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    sums=torch.zeros(222,device="cuda",dtype=torch.float64);counts=sums.clone()
    for r in fit:
        y,_=data.batch([r]);k,_=masks(y,60,90);c,m,_=observation(y,k,scale,bias)
        sums+=(c.double().square()*m).sum((0,1));counts+=m.sum((0,1))
    rms=(sums/counts.clamp(min=1)).sqrt()
    inv=torch.where(rms>1e-6,1/rms,torch.ones_like(rms)).float().cpu().numpy()
    np.save(out/"inv_rms.npy",inv)
    model,diffusion,_=load_model(assets);before=digest(model);bindings=[]
    chosen=[("fit",r) for r in select("fit",cfg["rollout_windows"])]+[("monitor",r) for r in select("monitor",cfg["rollout_monitor_windows"])]
    for i,(role,row) in enumerate(chosen):
        y,music=data.batch([row]);k,_=masks(y,60,90);states=[];times=[];raw=[]
        original=diffusion.model_predictions
        def record(x,cond,t,*args,**kwargs):
            step=record.step;record.step+=1
            value=original(x,cond,t,*args,**kwargs)
            if step in cfg["rollout_steps"]:
                states.append(x.detach().cpu().numpy().copy());times.append(int(t[0]));raw.append(value[1].detach().cpu().numpy().copy())
            return value
        record.step=0;diffusion.model_predictions=record
        torch.manual_seed(cfg["seed"]+900+i);torch.cuda.manual_seed_all(cfg["seed"]+900+i)
        sample,receipt=constrained_ddim(diffusion,music,y,k)
        diffusion.model_predictions=original
        p=out/(str(i)+".npz")
        np.savez_compressed(p,xt=np.concatenate(states),t=np.asarray(times),raw=np.concatenate(raw),sample=sample.cpu().numpy())
        bindings.append(dict(role=role,row=row,trajectory=ref(p),seed=cfg["seed"]+900+i,noise_sha256=receipt["noise_sha256"]))
        if replay:
            from dunhuang.research.source_tasks.common import same_arrays
            same_arrays(p,Path(replay)/p.name)
        print("R12_CACHE",i,role,flush=True)
    assert digest(model)==before
    result=dict(bindings=bindings,inv_rms=ref(out/"inv_rms.npy"),statistics_training_windows=len(fit),
        statistics_groups=groups["fit"],monitor_groups=groups["monitor"],new_neural_generations=len(bindings),
        optimizer_steps=0,backbone_digest=before,val_tensor_reads=0,operational_qualified=False)
    write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,capacity=ref(Path(capacity_root)/"result.json"),groups=groups,
        fit_rows=fit,feature_statistics_role="fit only",cache_semantics="actual B0 inpainting rollout x_t; hard 8+8 mask fixed; full 150-frame latent; not q_sample relabeling"))
    if replay:
        old=read(Path(replay)/"result.json")
        for key in ("statistics_training_windows","statistics_groups","monitor_groups","backbone_digest"):assert result[key]==old[key]
        assert np.array_equal(inv,np.load(checked(old["inv_rms"])))
        for a,b in zip(bindings,old["bindings"]):
            assert a["row"]==b["row"] and a["noise_sha256"]==b["noise_sha256"]

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["capacity","cache"]);p.add_argument("--config",required=True);p.add_argument("--capacity");p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    cfg=read(a.config)
    if a.phase=="capacity":capacity(cfg,a.out,a.replay_from)
    else:cache(cfg,a.capacity,a.out,a.replay_from)

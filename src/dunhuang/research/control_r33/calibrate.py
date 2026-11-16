"""Train-only task gradients and real finite AdamW steps; no heldout selection."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from .runtime import Runtime,save_change,restore_change
from .objective import partial_forward,error_scales
from dunhuang.research.reference_r26.geometry import fk22
from dunhuang.research.content_r32.common import local_metrics

def inputs(cfg,root):
    _,_,data,z=population_data();assert sha(root/"data/change_states.npz")==read(root/"data/protocol.json")["files"]["change_states.npz"]
    trace=np.load(root/"data/change_states.npz")
    selected=[1,2,4,5]
    states=dict(xt=trace["xt"][:,selected],times=trace["times"])
    body=np.stack([z["source139"][i,64:160,7:] for i in (0,1,0,1)])
    points=z["source22"][[0,1,0,1],64:160]-z["source22"][[0,1,0,1],64:160,:1]
    music=z["music"][[0,0,1,1]]
    packet=source(list(body),"change",uids=["female_FeiTian","male_pipa_1"]*2)
    return states,torch.tensor(body,device="cuda"),torch.tensor(points,device="cuda"),torch.tensor(music,device="cuda"),packet

def gradient_groups(module):
    names=[n for n,p in module.named_parameters()];params=list(module.parameters());offsets=np.r_[0,np.cumsum([p.numel() for p in params])]
    groups={"encoder":("parts","input","temporal","norm"),"injection":("q.","k.","v.","out.","qphase.")}
    slices={k:np.concatenate([np.arange(offsets[i],offsets[i+1]) for i,n in enumerate(names) if n.startswith(prefix)]) for k,prefix in groups.items()}
    return params,offsets,slices

def flat_gradient(loss,params,retain=True):
    torch.use_deterministic_algorithms(True)
    try:values=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
    finally:torch.use_deterministic_algorithms(False)
    return torch.cat([torch.zeros_like(p).flatten() if v is None else v.flatten() for p,v in zip(params,values)])

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    rt=Runtime(cfg);states,body,target,music,packet=inputs(cfg,root);cfg32=read("src/dunhuang/research/content_r32/config.json")
    params,offsets,groups=gradient_groups(rt.c.change);initial=digest(rt.c.change);arrays={};vectors={};scale_readouts=[];reports=[]
    for t in cfg["times"]:
        xt=torch.tensor(states["xt"][states["times"].tolist().index(t)],device="cuda");ts=torch.full((4,),t,device="cuda",dtype=torch.long)
        rt.c.clear_all();rt.c.set_source(packet)
        with torch.no_grad():
            pred=rt.d.model(xt,music,None,ts,cond_drop_prob=0);points=rt.f(rt.n.unnormalize(pred.clone()),rt.s)[:,64:160,:22]
            sv=error_scales(points,target,torch.ones_like(target[...,0],dtype=torch.bool)).cpu().numpy()
        arrays[f"scale_pred_{t}"]=pred.cpu().numpy();scale_readouts.append(sv)
    scales=np.maximum(np.median(np.stack(scale_readouts),axis=0),np.array(cfg["scale_floors"])[:,None]).tolist()
    for t in cfg["times"]:
        xt=torch.tensor(states["xt"][states["times"].tolist().index(t)],device="cuda")
        base,parts,dyn,dp,pred,fk,counts=partial_forward(rt,xt,music,packet,t,cfg["seed"]+1000+t,body,target,cfg32,scales)
        with torch.no_grad():official=rt.f(rt.n.unnormalize(pred.detach().clone()),rt.s)[:,:,:22]
        difference=float((official-fk.detach()).abs().max());assert difference<=2e-5
        gb=flat_gradient(base,params);gd=flat_gradient(dyn,params,False);vectors[f"base_{t}"]=gb.cpu().numpy();vectors[f"dyn_{t}"]=gd.cpu().numpy()
        g={}
        for name,indices in groups.items():
            x=gb[torch.tensor(indices,device="cuda")].double();y=gd[torch.tensor(indices,device="cuda")].double();nx=float(x.norm());ny=float(y.norm())
            assert min(nx,ny)>0 and np.isfinite([nx,ny]).all()
            g[name]=dict(base_norm=nx,dynamic_norm=ny,cosine=float((x@y)/(x.norm()*y.norm())))
        arrays[f"gradient_pred_{t}"]=pred.detach().cpu().numpy()
        reports.append(dict(t=t,base_loss=float(base.detach()),dynamic_loss=float(dyn.detach()),gradients=g,FK_max_difference_m=difference,
            Core_saturated_body_fraction=float((pred[:,64:160,7:].detach().abs()>1).float().mean()),template_counts=counts))
    weight=cfg["dynamic_gradient_fraction"]*min(float(np.median([r["gradients"][g]["base_norm"] for r in reports]))/float(np.median([r["gradients"][g]["dynamic_norm"] for r in reports])) for g in cfg["gradient_groups"])
    assert np.isfinite(weight) and weight>0
    observations=[];actual=[];checks=[];post_metrics=[];steps=0
    def state_loss(t,extra):
        xt=torch.tensor(states["xt"][states["times"].tolist().index(t)],device="cuda")
        base,parts,dyn,dp,pred,fk,counts=partial_forward(rt,xt,music,packet,t,cfg["seed"]+1000+t,body,target,cfg32,scales)
        loss=base+(weight*dyn if extra else 0)
        return loss,dict(base=float(base.detach()),dynamic=float(dyn.detach()),body=float(parts["body"].detach()),geometry=float(parts["local_geometry"].detach())),pred
    if not replay:
        (out/"snapshots").mkdir();save_change(rt,out/"initial.pt",cfg,"calibration",0)
    for t in cfg["times"]:
        for arm,extra in (("Extra",False),("Dyn",True)):
            stem=f"{arm}_{t}"
            restore_change(rt,(Path(replay) if replay else out)/"initial.pt");before=vector(rt.c.change).clone()
            if not replay:save_change(rt,out/f"snapshots/{stem}_pre.pt",cfg,arm,0)
            rt.c.change.zero_grad(set_to_none=True);loss,parts,pred=state_loss(t,extra);before_parts=parts
            torch.use_deterministic_algorithms(True)
            try:loss.backward()
            finally:torch.use_deterministic_algorithms(False)
            gradient=float(sum(p.grad.square().sum() for p in params if p.grad is not None).sqrt())
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if not replay:
                optimizer=torch.optim.AdamW(params,lr=cfg["learning_rate"]);optimizer.step();steps+=1;save_change(rt,out/f"snapshots/{stem}_post.pt",cfg,arm,1)
            else:restore_change(rt,Path(replay)/f"snapshots/{stem}_post.pt")
            current=dict(t=t,arm=arm,loss=float(loss.detach()),parts=parts,gradients={"all":gradient},update_norm=float((vector(rt.c.change)-before).norm()))
            if replay:
                old=read(Path(replay)/"result.json");expected=next(x for x in old["observations"] if x["t"]==t and x["arm"]==arm);checks.append(readout_check(current,expected));actual.append(current)
            observations.append(current)
            with torch.no_grad():post,post_parts,pred=state_loss(t,extra)
            arrays[f"finite_{stem}"]=pred.cpu().numpy()
            measures=[local_metrics(pred[i,64:160,7:],body[i].cpu().numpy(),target[i].cpu().numpy(),rt.s,rt.f) for i in range(4)]
            post_metrics.append(dict(t=t,arm=arm,pre=before_parts,post=post_parts,metrics=measures))
            rt.check()
    restore_change(rt,(Path(replay) if replay else out)/"initial.pt");assert digest(rt.c.change)==initial
    # Preserve the same-state candidate source geometry; no held generation is inspected.
    selected=[x for x in post_metrics if x["arm"]=="Dyn"]
    body_ratio=float(np.mean([x["post"]["body"] for x in selected]))/max(float(np.mean([x["pre"]["body"] for x in selected])),1e-12)
    geometry_ratio=float(np.mean([x["post"]["geometry"] for x in selected]))/max(float(np.mean([x["pre"]["geometry"] for x in selected])),1e-12)
    # Activity/range retention is measured below against the identical initial-state prediction.
    retention=[]
    for x in selected:
        t=x["t"];p=arrays[f"gradient_pred_{t}"]
        before=[local_metrics(p[i,64:160,7:],body[i].cpu().numpy(),target[i].cpu().numpy(),rt.s,rt.f) for i in range(4)]
        for prior,after in zip(before,x["metrics"]):
            for key in ("arm_range_ratio","body_activity_ratio"):
                retention.append(float(after[key]/prior[key]) if prior[key] is not None and prior[key]>1e-12 else 1.)
    rt.close()
    admitted=body_ratio<=1.1 and geometry_ratio<=1.1 and all(.8<=x<=1.25 for x in retention)
    result=dict(scales=scales,weight=weight,gradient_readouts=reports,finite_updates=post_metrics,observations=observations,actual_readouts=actual,replay_checks=checks,
        body_ratio=body_ratio,geometry_ratio=geometry_ratio,activity_extent_retention=retention,training_admitted=admitted,
        optimizer_steps=steps,sampler_calls=0,generated_trajectories=0,initial_change_digest=initial,heldout_used=False,
        calibration_scope="TRAIN_CONDITIONAL_TASK_GRADIENT_AND_EIGHT_INDEPENDENT_FINITE_STEPS_NOT_CFG_ROLLOUT_GRADIENT",
        gradient_vector_relative_l2_tolerance=1e-5,operational_qualified=False)
    np.savez_compressed(out/"arrays.npz",**arrays);np.savez_compressed(out/"gradients.npz",**vectors)
    if replay:
        prior=read(Path(replay)/"result.json");assert scales==prior["scales"] and weight==prior["weight"] and post_metrics==prior["finite_updates"] and admitted==prior["training_admitted"]
        same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
        saved=np.load(Path(replay)/"gradients.npz")
        errors={k:float(np.linalg.norm(v.astype(np.float64)-saved[k])/max(np.linalg.norm(saved[k].astype(np.float64)),1e-30)) for k,v in vectors.items()}
        assert all(x<=1e-5 for x in errors.values());result["gradient_replay_relative_l2"]=errors
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),files={p.name:sha(p) for p in out.glob("*.npz")}))
    print("R33_CALIBRATION",weight,admitted,body_ratio,geometry_ratio,steps,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

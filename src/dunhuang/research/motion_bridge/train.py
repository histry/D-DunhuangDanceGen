"""Fresh Bridge training: fixed final checkpoint, no DEV-based model selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import Data,load_model,digest,masks
from .study import population
from .model import MotionBridge,Injection,music_tokens,soft_control
from .objective import terms,total

def run(cfg,capacity,cache,out,arm,replay=None):
    out=fresh(out);cr=read(Path(capacity)/"result.json")
    if not cr["training_admitted"]:raise ValueError("No qualified Train-only injection direction")
    cached=read(Path(cache)/"result.json");protocol=read(Path(cache)/"protocol.json")
    assets,train,groups,select=population(cfg)
    assert groups==protocol["groups"]
    fit=protocol["fit_rows"];bindings=cached["bindings"];train_cache=[b for b in bindings if b["role"]=="fit"]
    monitor=[b for b in bindings if b["role"]=="monitor"]
    assert set(cached["statistics_groups"]).isdisjoint(cached["monitor_groups"])
    data=Data(np.load(checked(assets["scaler"])));model,diffusion,_=load_model(assets);before=digest(model)
    torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    adapter=MotionBridge(data.scale,data.bias,np.load(checked(cached["inv_rms"])),arm,cfg["perturbation_rms"]).cuda()
    hook=Injection(model,cr["selected_site"]);scale=adapter.scale;bias=adapter.bias
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arm=arm,capacity=ref(Path(capacity)/"result.json"),
        cache=ref(Path(cache)/"result.json"),fit_groups=groups["fit"],monitor_groups=groups["monitor"],selected_site=cr["selected_site"],
        training_windows=len(fit),final_checkpoint_only=True,validation_selection=False,val_tensor_reads=0,
        objective="raw x0 before hard overwrite: masked reconstruction + known Root/SO3 + BOTH seam D1/D2/relative SO3",
        contact_truth=False,synthetic_root_truth=False,baseline_c2_frozen=True))
    trajectory={b["trajectory"]["sha256"]:dict(np.load(checked(b["trajectory"]))) for b in bindings}
    def prediction(xt,music,t,value,k):
        return hook.predict(xt,music,t,adapter(value,k,xt,music_tokens(model,music),t))
    # Interface must learn from content, preserve empty mask and bound residual.
    y,m=data.batch([fit[0]]);k,_=masks(y,60,90);t=torch.tensor([519],device="cuda")
    d=adapter(y,k,y,music_tokens(model,m),t);assert torch.equal(d,torch.zeros_like(d))
    assert torch.equal(adapter(y,torch.zeros_like(k),y,music_tokens(model,m),t),torch.zeros_like(d))
    d.sum().backward();jac=float(adapter.output.weight.grad.norm())
    assert jac>0 and torch.isfinite(adapter.output.weight.grad).all();adapter.zero_grad()
    write(out/"interface_gate.json",dict(zero_initialized_identity=True,empty_context_exact_zero=True,head_content_jacobian=jac,quality_claimed=False))
    history=[];updates=0
    if replay:
        forbid_optimizers()
        payload=torch.load(checked(read(Path(replay)/"result.json")["checkpoint"]),map_location="cpu",weights_only=True)
        adapter.load_state_dict(payload["state"],strict=True)
    else:
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg["learning_rate"]);rng=np.random.default_rng(cfg["seed"])
        for step in range(cfg["training_steps"]):
            torch.manual_seed(cfg["seed"]+100000+step);torch.cuda.manual_seed_all(cfg["seed"]+100000+step)
            use_cache=bool(rng.random()<cfg["rollout_fraction"])
            if use_cache:
                chosen=[train_cache[int(i)] for i in rng.integers(0,len(train_cache),cfg["batch"])]
                target,music=data.batch([b["row"] for b in chosen]);zs=[];ts=[]
                for b in chosen:
                    z=trajectory[b["trajectory"]["sha256"]];j=int(rng.integers(0,len(z["t"])));zs.append(z["xt"][j]);ts.append(z["t"][j])
                xt=torch.tensor(np.stack(zs),device="cuda");t=torch.tensor(ts,device="cuda",dtype=torch.long)
            else:
                target,music=data.batch([fit[int(i)] for i in rng.integers(0,len(fit),cfg["batch"])])
                t=torch.tensor(rng.integers(0,1000,cfg["batch"]),device="cuda",dtype=torch.long);xt=diffusion.q_sample(target,t)
            k,_=masks(target,60,90);soft=k.clone()
            # Hard sampler/cache mask stays fixed. Soft-side dropout is explicitly supported.
            dropout=rng.random()
            if dropout<.15:soft[:,:75]=False
            elif dropout<.30:soft[:,75:]=False
            adapter.train();pred=prediction(xt,music,t,target,soft)
            parts=terms(pred,target,k,scale,bias);objective=total(parts,cfg["loss_weights"])
            assert torch.isfinite(objective)
            diagnostic={}
            if step in (0,49,199,cfg["training_steps"]-1):
                for name,value in parts.items():
                    grad=torch.autograd.grad(value,adapter.output.weight,retain_graph=True,allow_unused=True)[0]
                    diagnostic[name]=0. if grad is None else float(grad.norm())
            optimizer.zero_grad();objective.backward()
            assert all(p.grad is None for p in model.parameters())
            gn=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True));optimizer.step()
            history.append(dict(step=step,actual_rollout=use_cache,loss=float(objective.detach()),parts={k:float(v.detach()) for k,v in parts.items()},gradient_norm=gn,term_head_gradient=diagnostic))
            if (step+1)%25==0:print("R12_TRAIN",arm,step+1,history[-1],flush=True)
        updates=cfg["training_steps"]
    adapter.eval();scores=[];predictions={}
    with torch.no_grad():
        for i,b in enumerate(monitor):
            y,music=data.batch([b["row"]]);wrong,_=data.batch([monitor[(i+1)%len(monitor)]["row"]]);k,_=masks(y,60,90)
            z=trajectory[b["trajectory"]["sha256"]]
            for j,timestep in enumerate(z["t"]):
                xt=torch.tensor(z["xt"][j:j+1],device="cuda");t=torch.tensor([timestep],device="cuda",dtype=torch.long)
                for control in ("correct","removed","removed_left","removed_right","wrong_left","wrong_right"):
                    v,soft=soft_control(y,k,wrong,control);pred=prediction(xt,music,t,v,soft)
                    parts=terms(pred,y,k,scale,bias)
                    scores.append(dict(case=i,t=int(timestep),control=control,**{n:float(v) for n,v in parts.items()}))
                    predictions[str(len(scores))]=pred.cpu().numpy()
    np.savez_compressed(out/"monitor.npz",**predictions)
    if replay:
        old=read(Path(replay)/"result.json");assert old["monitor"]==scores
        from dunhuang.research.source_tasks.common import same_arrays
        same_arrays(out/"monitor.npz",Path(replay)/"monitor.npz");checkpoint=old["checkpoint"]
    else:
        torch.save(dict(schema="r12_explicit_motion_bridge_v1",arm=arm,site=cr["selected_site"],state=adapter.state_dict(),config=cfg,
            base_sha256=assets["checkpoint"]["sha256"],code_commit=commit(),capacity_sha256=ref(Path(capacity)/"result.json")["sha256"],
            cache_sha256=ref(Path(cache)/"result.json")["sha256"],operational_qualified=False),out/"bridge.pt");checkpoint=ref(out/"bridge.pt")
    assert digest(model)==before
    result=dict(arm=arm,checkpoint=checkpoint,parameters=sum(p.numel() for p in adapter.parameters()),optimizer_steps=updates,history=history,
        monitor=scores,monitor_groups=groups["monitor"],backbone_digest=before,backbone_unchanged=True,val_tensor_reads=0,
        new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);hook.close();print("R12_TRAIN_DONE",arm,updates,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--capacity",required=True);p.add_argument("--cache",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["B2","B3"],required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.capacity,a.cache,a.out,a.arm,a.replay_from)

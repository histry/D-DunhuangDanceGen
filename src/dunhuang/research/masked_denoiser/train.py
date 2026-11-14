"""Selected motion-denoiser finetuning; tiny fit precedes any population training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import masks
from dunhuang.research.motion_bridge.objective import terms,total
from .common import setup,make,tensor_hash,checkpoint_load
from .rollout import sample,CONTROLS

def run(cfg,out,stage="tiny",gate=None,replay=None):
    out=fresh(out)
    if stage=="main" and not read(Path(gate)/"result.json")["tiny_training_admitted"]:
        raise ValueError("Tiny Train model cannot control real contexts")
    assets,fit,tiny,monitor,cachefits,data,model,diffusion=setup(cfg)
    y,music=data.batch([b["row"] for b in tiny])
    frozen_before=None;torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    wrapper=make(cfg,model);frozen_before=wrapper.digest(True);initial=wrapper.digest()
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    known,_=masks(y,60,90);fixed=[]
    # Exact shared inputs/noise for the four-window mechanism exercise.
    for j,timestep in enumerate(cfg["timesteps"]):
        torch.manual_seed(cfg["seed"]+300+j);torch.cuda.manual_seed_all(cfg["seed"]+300+j)
        t=torch.full((4,),timestep,device="cuda",dtype=torch.long);fixed.append((diffusion.q_sample(y,t),t))
    with torch.no_grad():
        xt,t=fixed[0]
        reference=diffusion.model_predictions(xt,music,t,weight=2,clip_x_start=True)[1]
        empty=wrapper(xt,music,t,y,torch.zeros_like(known))
        assert torch.equal(reference,empty)
        poisoned=y.clone();poisoned[~known]=float("nan")
        assert torch.equal(wrapper(xt,music,t,y,known),wrapper(xt,music,t,poisoned,known))
    write(out/"interface.json",dict(real_edge_empty_mask_initialization_exact=True,hidden_observation_poison_exact=True,
        trainable_parameters=sum(p.numel() for p in wrapper.selected()),frozen_parameter_digest=frozen_before))
    initial_scores=[];initial_arrays={}
    with torch.no_grad():
        for j,(xt,t) in enumerate(fixed):
            p=wrapper(xt,music,t,y,known);initial_arrays[str(j)]=p.cpu().numpy()
            for i in range(4):initial_scores.append(dict(case=i,t=int(t[i]),**{k:float(v) for k,v in terms(p[i:i+1],y[i:i+1],known[i:i+1],scale,bias).items()}))
    np.savez_compressed(out/"initial.npz",**initial_arrays)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,stage=stage,fit_population=fit if stage=="main" else [b["row"] for b in tiny],
        monitor_population=[b["row"] for b in monitor],trainable_names=wrapper.trainable_names,
        teacher_distillation=False,teacher_note="No physically qualified motion teacher; direct paired target training is independent",
        base_checkpoint=assets["checkpoint"],music_encoder_frozen=True,motion_context_in_both_cfg_branches=True,
        new_noise_schedule=False,validation_model_selection=False,final_checkpoint_only=True))
    history=[];ledger=[];updates=0;self_caches=[];cache=[];self_snapshots=[]
    if replay:
        forbid_optimizers();old=read(Path(replay)/"result.json");checkpoint_load(cfg,wrapper,old["checkpoint"],assets)
        if stage=="main":
            from .rollout import replay_arrays
            for snapshot in old["self_snapshots"]:
                checkpoint_load(cfg,wrapper,snapshot["checkpoint"],assets)
                assert wrapper.digest()==snapshot["model_digest"]
                for b in [b for b in old["self_caches"] if b["update"]==snapshot["update"]]:
                    yy,mm=data.batch([b["row"]]);i=b["case"];wrong,_=data.batch([cachefits[(i+1)%8]["row"]])
                    folder=out/("self_"+str(b["update"])+"_"+str(i))
                    sample(diffusion,wrapper,yy,mm,wrong,cfg["seed"]+10000+b["update"]*10+i,cfg,folder,counterfactuals=False)
                    replay_arrays(folder,Path(checked(b["trace"])).parent)
            checkpoint_load(cfg,wrapper,old["checkpoint"],assets)
    else:
        opt=torch.optim.AdamW(wrapper.selected(),lr=cfg["learning_rate"]);rng=np.random.default_rng(cfg["seed"])
        count=cfg["tiny_steps"] if stage=="tiny" else cfg["main_steps"]
        for step in range(count):
            if stage=="main" and step in cfg["self_cache_updates"]:
                snapshot=wrapper.digest();cache=[]
                snapshot_file=out/("self_state_"+str(step)+".pt")
                torch.save(dict(schema=cfg["schema"],config=cfg,state=wrapper.state(),base_sha256=assets["checkpoint"]["sha256"],
                    code_commit=commit(),producer_update=step,operational_qualified=False),snapshot_file)
                self_snapshots.append(dict(update=step,model_digest=snapshot,checkpoint=ref(snapshot_file)))
                for i,b in enumerate(cachefits[:cfg["self_cache_windows"]]):
                    yy,mm=data.batch([b["row"]]);wrong,_=data.batch([cachefits[(i+1)%8]["row"]])
                    folder=out/("self_"+str(step)+"_"+str(i))
                    sample(diffusion,wrapper,yy,mm,wrong,cfg["seed"]+10000+step*10+i,cfg,folder,counterfactuals=False)
                    z=dict(np.load(folder/"trace.npz"));cache.append((b["row"],z));self_caches.append(dict(update=step,case=i,row=b["row"],model_digest=snapshot,trace=ref(folder/"trace.npz")))
                assert wrapper.digest()==snapshot
            torch.manual_seed(cfg["seed"]+100000+step);torch.cuda.manual_seed_all(cfg["seed"]+100000+step)
            use_cache=stage=="main" and bool(cache) and rng.random()<cfg["self_cache_fraction"]
            if stage=="tiny":
                target=y;mm=music;xt,t=fixed[step%4];identity=[b["row"] for b in tiny]
            elif use_cache:
                chosen=[cache[int(i)] for i in rng.integers(0,len(cache),cfg["main_batch"])]
                identity=[r for r,z in chosen];target,mm=data.batch(identity);zs=[];ts=[]
                for row,z in chosen:
                    j=int(rng.integers(0,len(z["t"])));zs.append(z["xt"][j]);ts.append(z["t"][j])
                xt=torch.tensor(np.stack(zs),device="cuda");t=torch.tensor(ts,device="cuda",dtype=torch.long)
            else:
                identity=[fit[int(i)] for i in rng.integers(0,len(fit),cfg["main_batch"])]
                target,mm=data.batch(identity);t=torch.tensor(rng.integers(0,1000,len(target)),device="cuda",dtype=torch.long);xt=diffusion.q_sample(target,t)
            k,_=masks(target,60,90);pred=wrapper(xt,mm,t,target,k);parts=terms(pred,target,k,scale,bias);objective=total(parts,cfg["loss_weights"])
            if not torch.isfinite(objective):raise ValueError("Nonfinite R13 direct-denoiser loss")
            opt.zero_grad();objective.backward()
            assert all(p.grad is None for p in wrapper.net.parameters() if not p.requires_grad)
            module_gradients={}
            if step in (0,count-1):
                for prefix in cfg["trainable_prefixes"]:
                    values=[p.grad.detach().square().sum() for n,p in wrapper.net.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    module_gradients[prefix]=float(torch.stack(values).sum().sqrt()) if values else 0.
            grad=float(torch.nn.utils.clip_grad_norm_(wrapper.selected(),1.,error_if_nonfinite=True));opt.step()
            history.append(dict(step=step,loss=float(objective.detach()),parts={n:float(v.detach()) for n,v in parts.items()},gradient_norm=grad,module_gradient_norm=module_gradients))
            ledger.append(dict(step=step,rows=[dict(raw_sha256=r["raw_motion"]["sha256"],start=r["start_frame"],group=r["source_group"]) for r in identity],
                x_t_sha256=tensor_hash(xt),t=t.cpu().tolist(),observed_sha256=tensor_hash(target[k]),self_cache=use_cache))
            if (step+1)%20==0:print("R13_TRAIN",stage,step+1,history[-1],flush=True)
        updates=count
    predictions={};scores=[]
    with torch.no_grad():
        for j,(xt,t) in enumerate(fixed):
            for control in CONTROLS:
                from dunhuang.research.motion_bridge.model import soft_control
                v,k=soft_control(y,known,y.roll(1,0),control);p=wrapper(xt,music,t,v,k);predictions[f"{j}_{control}"]=p.cpu().numpy()
                for i in range(4):
                    scores.append(dict(case=i,t=int(t[i]),control=control,**{n:float(v) for n,v in terms(p[i:i+1],y[i:i+1],known[i:i+1],scale,bias).items()}))
    np.savez_compressed(out/"q_eval.npz",**predictions)
    if replay:
        from dunhuang.research.source_tasks.common import same_arrays
        assert scores==old["q_scores"] and initial_scores==old["initial_scores"]
        same_arrays(out/"q_eval.npz",Path(replay)/"q_eval.npz");same_arrays(out/"initial.npz",Path(replay)/"initial.npz");checkpoint=old["checkpoint"]
    else:
        torch.save(dict(schema=cfg["schema"],config=cfg,state=wrapper.state(),code_commit=commit(),stage=stage,
            base_sha256=assets["checkpoint"]["sha256"],trainable_names=wrapper.trainable_names,operational_qualified=False),out/"denoiser.pt")
        checkpoint=ref(out/"denoiser.pt")
    assert wrapper.digest(True)==frozen_before
    result=dict(checkpoint=checkpoint,stage=stage,optimizer_steps=updates,initial_digest=initial,final_digest=wrapper.digest(),frozen_digest=frozen_before,
        initial_scores=initial_scores,q_scores=scores,history=history,ledger=ledger,self_caches=self_caches,self_snapshots=self_snapshots,
        parameters=sum(p.numel() for p in wrapper.selected()),music_encoder_frozen=True,selected_motion_backbone_updated=wrapper.digest()!=initial,
        new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);print("R13_TRAIN_DONE",stage,updates,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--stage",choices=["tiny","main"],default="tiny");p.add_argument("--gate");p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.stage,a.gate,a.replay_from)

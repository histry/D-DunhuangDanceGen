"""Frozen R12 direction teacher on actual B0 states and complete teacher rollouts."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import masks,digest
from dunhuang.research.motion_bridge.model import Injection
from dunhuang.sampling import constrained_ddim
from .common import setup,tensor_hash
from .model import known_loss

def direction(hook,xt,music,t,value,k,scale,bias,radius):
    with torch.enable_grad():
        delta=torch.zeros(len(xt),150,512,device=xt.device,requires_grad=True)
        pred=hook.predict(xt.detach(),music,t,delta)
        objective=known_loss(pred,value,k,scale,bias)
        grad=torch.autograd.grad(objective,delta)[0]
        if not torch.isfinite(grad).all():raise ValueError("Nonfinite teacher direction")
        rms=grad.square().mean((1,2),keepdim=True).clamp(min=1e-24).sqrt()
        return (-radius*grad/rms).detach()

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();assets,_,tiny,_,_,data,model,diffusion=setup(cfg);before=digest(model)
    y,music=data.batch([b["row"] for b in tiny]);known,_=masks(y,60,90)
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    hook=Injection(model,cfg["teacher_site"]);arrays={};records=[]
    caches=[dict(np.load(checked(b["trajectory"]))) for b in tiny]
    for j,timestep in enumerate(cfg["timesteps"]):
        assert all(int(z["t"][j])==timestep for z in caches)
        xt=torch.tensor(np.stack([z["xt"][j] for z in caches]),device="cuda");t=torch.full((4,),timestep,device="cuda",dtype=torch.long)
        d=direction(hook,xt,music,t,y,known,scale,bias,cfg["teacher_rms"])
        poison=y.clone();poison[~known]=float("nan")
        assert torch.equal(d,direction(hook,xt,music,t,poison,known,scale,bias,cfg["teacher_rms"]))
        wrong=direction(hook,xt,music,t,y.roll(1,0),known,scale,bias,cfg["teacher_rms"])
        torch.manual_seed(cfg["seed"]+j);torch.cuda.manual_seed_all(cfg["seed"]+j)
        random=torch.randn_like(d);random=random/random.square().mean((1,2),keepdim=True).sqrt()*cfg["teacher_rms"]
        for control,delta in (("B0",None),("correct",d),("wrong",wrong),("random",random)):
            with torch.no_grad():pred=hook.predict(xt,music,t,delta)
            for i in range(4):
                arrays[f"{j}_{i}_{control}"]=pred[i].cpu().numpy()
                records.append(dict(case=i,t=timestep,control=control,known_objective=float(known_loss(pred[i:i+1],y[i:i+1],known[i:i+1],scale,bias)),
                    direction_rms=0. if delta is None else float(delta[i].square().mean().sqrt()),x_t_sha256=tensor_hash(xt[i])))
    np.savez_compressed(out/"probes.npz",**arrays);bindings=[]
    original=diffusion.model_predictions
    for i,b in enumerate(tiny):
        value=y[i:i+1];m=music[i:i+1];k=known[i:i+1];raw=[]
        def predict(x,c,t,*args,**kwargs):
            rng=torch.cuda.get_rng_state()
            d=direction(hook,x,c,t,value,k,scale,bias,cfg["teacher_rms"])
            with torch.no_grad():p=hook.predict(x,c,t,d)
            assert torch.equal(rng,torch.cuda.get_rng_state());raw.append(p.cpu().numpy().copy())
            return diffusion.predict_noise_from_start(x,t,p),p
        diffusion.model_predictions=predict
        torch.manual_seed(b["seed"]);torch.cuda.manual_seed_all(b["seed"])
        sample,receipt=constrained_ddim(diffusion,m,value,k);diffusion.model_predictions=original
        assert receipt["noise_sha256"]==b["noise_sha256"]
        f=out/(str(i)+".npz");np.savez_compressed(f,sample=sample.cpu().numpy(),raw=np.concatenate(raw),baseline=caches[i]["sample"])
        bindings.append(dict(case=i,row=b["row"],rollout=ref(f),seed=b["seed"],noise_sha256=receipt["noise_sha256"]))
        if replay:
            from dunhuang.research.source_tasks.common import same_arrays
            same_arrays(f,Path(replay)/f.name)
        print("R13_TEACHER_ROLLOUT",i,flush=True)
    hook.close();assert digest(model)==before
    result=dict(records=records,bindings=bindings,optimizer_steps=0,hidden_target_poison_exact=True,backbone_digest=before,
        new_neural_generations=4,operational_qualified=False,teacher_scope="known-only feature direction; physical qualification pending")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,assets=ref(checked(read(Path(cfg["preflight"])/"protocol.json")["assets"])),
        r12_cache=ref(Path(cfg["r12_root"])/"cache/result.json"),unknown_target_used_for_direction=False,delta_optimizer_steps=0,teacher_method="one RMS-normalized gradient step, not an argmin solver"))
    if replay:
        old=read(Path(replay)/"result.json");assert records==old["records"]
        from dunhuang.research.source_tasks.common import same_arrays
        same_arrays(out/"probes.npz",Path(replay)/"probes.npz")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)

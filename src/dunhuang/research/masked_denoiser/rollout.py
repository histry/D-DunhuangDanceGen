"""Common 50-step research sampler; correct/CF conditions never change hard anchors."""
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write
from dunhuang.sampling import constrained_ddim
from dunhuang.research.boundary_context.common import masks
from dunhuang.research.motion_bridge.model import soft_control
from .common import tensor_hash

CONTROLS=("correct","removed","removed_left","removed_right","wrong_left","wrong_right")

def sample(diffusion,predict,y,music,wrong,seed,cfg,out,*,driver="correct",counterfactuals=True):
    out=Path(out);out.mkdir(exist_ok=False);hard,_=masks(y,60,90);original=diffusion.model_predictions
    states=[];predictions=[];times=[];raw=[];steps=[];step=0
    def forward(x,c,t,*args,**kwargs):
        nonlocal step
        rng=torch.cuda.get_rng_state();value,k=soft_control(y,hard,wrong,driver)
        p=predict(x,c,t,value,k);base=p
        indices=cfg["self_cache_steps"]
        if step in indices:
            controls=CONTROLS if counterfactuals and driver=="correct" else (driver,)
            pp=[]
            for control in controls:
                if control==driver:q=base
                else:
                    v,m=soft_control(y,hard,wrong,control);q=predict(x,c,t,v,m)
                pp.append(q.cpu().numpy().copy())
            states.append(x.cpu().numpy().copy());times.append(int(t[0]));predictions.append(np.concatenate(pp))
        raw.append(base.cpu().numpy().copy());steps.append(dict(step=step,t=int(t[0]),x_t_sha256=tensor_hash(x)))
        step+=1;assert torch.equal(rng,torch.cuda.get_rng_state())
        return diffusion.predict_noise_from_start(x,t,base),base
    diffusion.model_predictions=forward
    try:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        with torch.no_grad():motion,receipt=constrained_ddim(diffusion,music,y,hard)
    finally:diffusion.model_predictions=original
    np.savez_compressed(out/"trace.npz",xt=np.concatenate(states),t=np.asarray(times),prediction=np.stack(predictions),raw=np.concatenate(raw),sample=motion.cpu().numpy())
    write(out/"sampler.json",dict(receipt,driver=driver,conditional_probe_steps=cfg["self_cache_steps"],steps=steps))
    return dict(sample=motion.detach(),receipt=receipt)

def replay_arrays(folder,other):
    from dunhuang.research.source_tasks.common import same_arrays
    from dunhuang.research.source_capacity.common import read
    folder=Path(folder);other=Path(other);same_arrays(folder/"trace.npz",other/"trace.npz")
    assert read(folder/"sampler.json")==read(other/"sampler.json")

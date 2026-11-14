"""Aligned unknown/observed Root/rotation readouts, stratified by diffusion time."""
import torch
from .common import state,coverage
from .objective import reconstruction,geometry

def run(cfg,cache,predict,head):
    arrays={};records=[]
    with torch.no_grad():
        for case in cfg["cases"]:
            for t in cfg["timesteps"]:
                s=state(cfg,cache,case,t,True,cfg["seed"]+50000+case*100+t)
                y,m,o,v,k,h,meta,ts,xt=s
                for control in cfg["controls"]:
                    p,_=predict(xt,m,v,k,meta,ts,control);terms,counts=reconstruction(p,y,k)
                    name=f"{case}_{t}_{control}";arrays[name]=p[0].cpu().numpy()
                    records.append(dict(case=case,t=t,control=control,losses={n:float(v) for n,v in terms.items()},
                        counts=counts,geometry=geometry(p,cache["scale"],cache["bias"]),
                        state_support=coverage(k,head.memory_mode),
                        alpha=float(cache["diffusion"].alphas_cumprod[t]),
                        raw_observed_root_rmse_m=float(((p[:,:,4:7]-y[:,:,4:7])/cache["scale"][4:7])[k[:,:,4:7]].square().mean().sqrt())))
    return arrays,records

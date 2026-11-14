import sys
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,pinned,write
from dunhuang.research.source_capacity.common import read
from .balanced import make

def main():
    q=read(sys.argv[1])
    if q.get("research_r10") is not True or q["mode"]!="anchored":raise ValueError("Explicit R10 contextual research required")
    sys.path.insert(0,str(pinned(q["edge_root"],q["edge_commit"])))
    import dunhuang.edge_backend as backend
    context=np.load(checked(q["context"]));backend.anchor_windows=lambda frames,anchors:(context["values"].copy(),context["mask"].copy())
    holder={};normal=backend.normalize_constraints;counter=0
    def normalize(norm,values,mask):
        value,known=normal(norm,values,mask);holder.update(value=value.cuda(),known=known.cuda(),scale=norm.scaler.scale_.cuda(),bias=norm.scaler.min_.cuda())
        return value,known
    backend.normalize_constraints=normalize
    if q.get("r10_adapter"):
        payload=torch.load(checked(q["r10_adapter"]),map_location="cpu",weights_only=True)
        if payload["base_sha256"]!=q["checkpoint"]["sha256"]:raise ValueError("Context model lineage differs")
        scaler=np.load(checked(q["scaler"]));adapter=make(payload,scaler["scale"],scaler["bias"]).cuda().eval();adapter.load_state_dict(payload["state"],strict=True)
        from model.diffusion import GaussianDiffusion
        init=GaussianDiffusion.__init__;prediction=GaussianDiffusion.model_predictions
        def attach(self,*args,**kwargs):
            init(self,*args,**kwargs)
            def condition(module,inputs,music):
                v=holder["value"];k=holder["known"]
                if q.get("soft_control")=="removed":k=torch.zeros_like(k)
                if q.get("soft_control")=="wrong":
                    alt=np.load(checked(q["wrong_context"]))["values"]
                    v=torch.tensor(alt,device="cuda")*holder["scale"]+holder["bias"]
                return music+adapter(v,k,music,holder["t"])
            args[0].cond_encoder.register_forward_hook(condition)
        def predict(self,x,music,t,*args,**kwargs):
            nonlocal counter
            holder["t"]=t;counter+=1;return prediction(self,x,music,t,*args,**kwargs)
        GaussianDiffusion.__init__=attach;GaussianDiffusion.model_predictions=predict
    backend.main()
    if q.get("r10_adapter") and counter!=50:raise ValueError("Sampler timestep injection did not cover all 50 steps")
    p=Path(q["output"])/"generation.json";g=read(p);g.update(r10_adapter=q.get("r10_adapter"),soft_control=q.get("soft_control"),explicit_timestep_steps=counter,
        root_origin_native=q["root_origin_native"],context_interface="content-centered actual state; Root-relative and SO3; correct hard context for all controls",operational_qualified=False)
    write(p,g)
if __name__=="__main__":main()

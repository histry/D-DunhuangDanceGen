"""Actual masked conditional inpainting using the unchanged single EDGE backend."""
import sys
from pathlib import Path
import torch
from dunhuang.io import checked,write,pinned
from dunhuang.research.source_capacity.common import read
from .adapter import BoundaryContextAdapter
def main():
    q=read(sys.argv[1])
    if q.get("research_b2") is not True or q["mode"]!="anchored":raise ValueError("Explicit B2 research request required")
    sys.path.insert(0,str(pinned(q["edge_root"],q["edge_commit"])))
    import numpy as np
    import dunhuang.edge_backend as backend
    context=np.load(checked(q["context"]));backend.anchor_windows=lambda frames,anchors:(context["values"].copy(),context["mask"].copy())
    holder={};original=backend.normalize_constraints
    def normalize(norm,v,m):
        value,known=original(norm,v,m);holder.update(value=value.cuda(),known=known.cuda());return value,known
    backend.normalize_constraints=normalize
    if q.get("b2_adapter"):
        payload=torch.load(checked(q["b2_adapter"]),map_location="cpu",weights_only=True)
        if payload["schema"]!="boundary_context_adapter_v1" or payload["base_sha256"]!=q["checkpoint"]["sha256"]:raise ValueError("Bridge checkpoint lineage differs")
        adapter=BoundaryContextAdapter().cuda().eval();adapter.load_state_dict(payload["state"],strict=True)
        from model.diffusion import GaussianDiffusion
        init=GaussianDiffusion.__init__
        def attach(self,*args,**kwargs):
            init(self,*args,**kwargs)
            args[0].cond_encoder.register_forward_hook(lambda module,inputs,music:music+adapter(holder["value"],holder["known"],music))
        GaussianDiffusion.__init__=attach
    backend.main()
    file=Path(q["output"])/"generation.json";g=read(file);g.update(research_boundary_adapter=q.get("b2_adapter"),research_unqualified_adapter=bool(q.get("b2_adapter")),adapter_music_qualified=False,full_motion_qualified=False)
    write(file,g)
if __name__=="__main__":main()

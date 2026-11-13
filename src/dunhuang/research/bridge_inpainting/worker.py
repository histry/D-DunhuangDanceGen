"""Dual-context DDIM inpainting, sharing the frozen EDGE denoiser and hard-mask logic."""
import sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,pinned,write
from dunhuang.research.source_capacity.common import read

def main():
    q=read(sys.argv[1])
    if q.get("research_bridge_v1") is not True or q["mode"]!="anchored" or q["frames"]!=150 or q["windows"]!=1:raise ValueError("Explicit local Bridge request required")
    c=np.load(checked(q["bridge_context"]));values=c["values"];mask=c["mask"]
    if values.shape!=(1,150,151) or mask.shape!=values.shape or mask.dtype!=bool or mask[:,:,:4].any() or not mask[:,:,4:7].any():raise ValueError("Actual generated Root/rotation context required")
    root=pinned(q["edge_root"],q["edge_commit"]);sys.path.insert(0,str(root))
    import dunhuang.edge_backend as backend
    backend.anchor_windows=lambda frames,anchors:(values.copy(),mask.copy())
    original=backend.normalize_constraints;holder={};counter=0
    def normalize(normalizer,v,m):
        result=original(normalizer,v,m) # strict support and roundtrip; no clipping
        if q["bridge_source_condition"]:
            from dunhuang.research.primitive_reperformance.guide import prepare
            holder["guide"]=prepare(q,normalizer,Path(q["output"]),"cuda")
        return result
    backend.normalize_constraints=normalize
    from model.diffusion import GaussianDiffusion
    prediction=GaussianDiffusion.model_predictions
    def predictions(self,*args,**kwargs):
        nonlocal counter
        result=prediction(self,*args,**kwargs);counter+=1
        if q["bridge_source_condition"]:
            items=list(result);items[1]=holder["guide"](items[1],counter/50)
            result=type(result)(*items) if hasattr(result,"_fields") else tuple(items)
        return result
    GaussianDiffusion.model_predictions=predictions
    backend.main()
    # Frozen affine normalizer identity is checked by the backend; the final
    # hard-mask step restores clean known context, unlike upstream i=0 x fallback.
    saved=read(Path(q["output"])/"generation.json")
    if counter!=50:raise ValueError("Bridge reverse-step count differs")
    write(Path(q["output"])/"bridge_sampling.json",dict(steps=counter,denoiser="same frozen EDGE checkpoint",known_context="actual saved output, Root and full rotations",
        source_condition=q["bridge_source_condition"],unknown_edit=q["edit_interval"],
        hard_context_restored_each_reverse_step=True,final_clean_context_restored=True,no_context_clipping=True,
        stochastic_noise_sha256=saved["sampler"]["noise_sha256"],new_neural_weight_updates=0,
        method="project constrained_ddim hard-mask inpainting, 50 steps; not an exact upstream 1000-step DDPM reproduction"))
if __name__=="__main__":main()

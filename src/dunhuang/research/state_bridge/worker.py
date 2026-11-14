"""A0-only sampler plus shared-state soft interventions; frozen model/CFG/noise."""
import sys,hashlib
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,pinned,write
from dunhuang.research.source_capacity.common import read,commit
from dunhuang.research.condition_dependency.balanced import BalancedContextAdapter
from .control import SideDiagnosticAdapter,soft_control

def main():
    q=read(sys.argv[1])
    if q.get("research_r11") is not True or q["mode"]!="anchored":raise ValueError("Explicit R11 frozen diagnostic required")
    sys.path.insert(0,str(pinned(q["edge_root"],q["edge_commit"])))
    import dunhuang.edge_backend as backend
    context=np.load(checked(q["context"]));backend.anchor_windows=lambda frames,anchors:(context["values"].copy(),context["mask"].copy())
    holder={};normal=backend.normalize_constraints;rows=[];predictions=[];latents=[]
    def normalize(norm,values,mask):
        v,k=normal(norm,values,mask)
        holder.update(value=v.cuda(),known=k.cuda())
        return v,k
    backend.normalize_constraints=normalize
    payload=torch.load(checked(q["r10_adapter"]),map_location="cpu",weights_only=True)
    if payload["schema"]!="r10_balanced_context_v1" or payload["base_sha256"]!=q["checkpoint"]["sha256"]:raise ValueError("Frozen C2 lineage required")
    s=np.load(checked(q["scaler"]));adapter=SideDiagnosticAdapter(s["scale"],s["bias"],payload["state"]["feature_inv_rms"]).cuda().eval()
    adapter.load_state_dict(payload["state"],strict=True)
    for p in adapter.parameters():p.requires_grad_(False)
    wrong=np.load(checked(q["r11_wrong_context"]))["values"]
    wrong=torch.tensor(wrong,device="cuda")*adapter.scale+adapter.bias
    from model.diffusion import GaussianDiffusion
    init=GaussianDiffusion.__init__;prediction=GaussianDiffusion.model_predictions
    def attach(self,*args,**kwargs):
        init(self,*args,**kwargs)
        def condition(module,args,music):
            v,k=soft_control(holder["value"],holder["known"],wrong,holder["arm"])
            residual=adapter(v,k,music,holder["t"])
            holder["residuals"].append(float(residual.square().mean().sqrt()))
            return music+residual
        args[0].cond_encoder.register_forward_hook(condition)
    def predict(self,x,music,t,*args,**kwargs):
        holder.update(t=t,arm="A0",residuals=[])
        adapter.hard_anchor=holder["value"][:,59:60,4:6].clone()
        parent,_=BalancedContextAdapter.features(adapter,holder["value"],holder["known"],t)
        actual,_=adapter.features(holder["value"],holder["known"],t)
        assert torch.equal(parent,actual)
        cpu_rng=torch.random.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
        before=x.detach().cpu().contiguous().numpy();xhash=hashlib.sha256(before.tobytes()).hexdigest()
        all_pred=[];measures=[];original=None
        for arm in q["r11_arms"]:
            holder.update(arm=arm,residuals=[])
            value=prediction(self,x,music,t,*args,**kwargs)
            if arm=="A0":original=value
            v=value[1]
            if arm=="A5":assert all(z==0 for z in holder["residuals"])
            all_pred.append(v[0,50:100].detach().cpu().numpy())
            measures.append(dict(arm=arm,residual_rms_by_branch=holder["residuals"].copy(),
                raw_known_error_max=float(abs(v[holder["known"]]-holder["value"][holder["known"]]).max())))
        assert torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        assert hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()==xhash
        holder["arm"]="A0"
        predictions.append(np.stack(all_pred));latents.append(before[0,50:100].copy())
        rows.append(dict(step=len(rows),t=int(t[0]),x_t_sha256=xhash,rng_unchanged=True,a0_features_exact=True,arms=measures))
        return original
    GaussianDiffusion.__init__=attach;GaussianDiffusion.model_predictions=predict
    backend.main()
    assert len(rows)==50
    out=Path(q["output"]);np.savez_compressed(out/"trace.npz",prediction=np.stack(predictions),latent=np.stack(latents))
    write(out/"trace.json",dict(steps=rows,roi=[50,100],arms=q["r11_arms"],control_kind="shared A0 x_t forward intervention, not independent counterfactual rollouts"))
    g=read(out/"generation.json");g.update(code_commit=commit(),research_r11=True,reference_adapter=q["r10_adapter"],operational_qualified=False,
        hard_context_unchanged=True,common_gauge="actual hard frame59 horizontal Root, independent of soft observation mask",sampler_driving_arm="A0")
    write(out/"generation.json",g)
if __name__=="__main__":main()

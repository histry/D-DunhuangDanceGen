"""Read-only Train diagnosis of representability, cap usage and raw-anchor tradeoffs.

No new model selection or optimizer; four monitor music groups remain the unit.
Gradient magnitudes are diagnostic, not causal attribution.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import Data,load_model,digest
from .study import population
from .model import MotionBridge,music_tokens

def run(cfg,root,out,replay=None):
    out=fresh(out);forbid_optimizers();root=Path(root)
    assets,_,groups,_=population(cfg);cache=read(root/"cache/result.json");cp=read(root/"cache/protocol.json")
    export=np.load(checked(assets["scaler"]));data=Data(export);support={}
    for role,rows in (("fit",cp["fit_rows"]),("monitor",[b["row"] for b in cache["bindings"] if b["role"]=="monitor"])):
        root_bad=0;pose_bad=0;lower=[]
        for row in rows:
            native=data.native(row);origin=native[0,4:7].copy();origin[1]=0
            z=MotionAdapter.to_edge(native,origin)*data.scale+data.bias
            inds=np.r_[52:60,90:98];known=z[inds]
            root_bad+=int((abs(known[:,4:7])>1+2e-6).any())
            pose_bad+=int((abs(known[:,7:])>1+2e-6).any())
            lo=(np.clip(known[:,4:7],-1,1)-known[:,4:7])/data.scale[4:7]
            lower.append(float(np.sqrt(np.mean(lo**2))))
        support[role]=dict(windows=len(rows),known_root_outside_clamp_windows=root_bad,known_rotation_outside_clamp_windows=pose_bad,
            root_unreachable_lower_bound_rmse_mean_m=float(np.mean(lower)),root_unreachable_lower_bound_rmse_max_m=max(lower))
    model,_,_=load_model(assets);before=digest(model)
    monitor=[b for b in cache["bindings"] if b["role"]=="monitor"]
    known=np.r_[52:60,90:98];results={};baseline=[]
    def root_error(p,y):
        raw=(p[:,:,4:7]-y[:,:,4:7])/data.scale[4:7]
        return dict(known_root_rmse_m=float(np.sqrt(np.mean(raw[:,known]**2))),
            unknown_root_rmse_m=float(np.sqrt(np.mean(raw[:,60:90]**2))))
    for arm in ("B3","B2"):
        result=read(root/arm/"result.json");payload=torch.load(checked(result["checkpoint"]),map_location="cpu",weights_only=True)
        assert payload["config"]==cfg and payload["base_sha256"]==assets["checkpoint"]["sha256"]
        adapter=MotionBridge(data.scale,data.bias,payload["state"]["inv_rms"],arm,cfg["perturbation_rms"]).cuda().eval()
        adapter.load_state_dict(payload["state"],strict=True);saved=np.load(root/arm/"monitor.npz");states=[]
        with torch.no_grad():
            for i,b in enumerate(monitor):
                y,music=data.batch([b["row"]]);tokens=music_tokens(model,music);target=y.cpu().numpy()
                mask=torch.zeros_like(y,dtype=torch.bool);mask[:,known,4:]=True
                z=np.load(checked(b["trajectory"]))
                for j,timestep in enumerate(z["t"]):
                    index=i*24+j*6+1
                    score=next(s for s in result["monitor"] if s["case"]==i and s["t"]==int(timestep) and s["control"]=="correct")
                    xt=torch.tensor(z["xt"][j:j+1],device="cuda");t=torch.tensor([timestep],device="cuda",dtype=torch.long)
                    capped=adapter(y,mask,xt,tokens,t)
                    adapter.cap=1e6;uncapped=adapter(y,mask,xt,tokens,t);adapter.cap=cfg["perturbation_rms"]
                    states.append(dict(case=i,t=int(timestep),**root_error(saved[str(index)],target),
                        residual_rms=float(capped.square().mean().sqrt()),uncapped_residual_rms=float(uncapped.square().mean().sqrt()),
                        cap_active=bool(uncapped.square().mean().sqrt()>cfg["perturbation_rms"]),raw_terms=score))
                    if arm=="B3":baseline.append(dict(case=i,t=int(timestep),**root_error(z["raw"][j:j+1],target)))
        gradients=[]
        for h in result["history"]:
            if h["term_head_gradient"]:
                gradients.append(dict(step=h["step"],weighted_head_gradient_norm={k:v*cfg["loss_weights"][k] for k,v in h["term_head_gradient"].items()}))
        results[arm]=dict(states=states,cap_active_states=sum(s["cap_active"] for s in states),total_states=len(states),
            known_root_rmse_mean_m=float(np.mean([s["known_root_rmse_m"] for s in states])),
            unknown_root_rmse_mean_m=float(np.mean([s["unknown_root_rmse_m"] for s in states])),gradient_magnitudes=gradients)
    assert digest(model)==before
    result=dict(support=support,monitor_baseline=baseline,arms=results,monitor_music_groups=groups["monitor"],
        config=cfg,optimizer_steps=0,backbone_unchanged=True,operational_qualified=False,
        limitations=["Train-only diagnostic, not DEV tuning","head gradient norms do not identify causal loss effects",
            "cap saturation is evidence at saved B0 off-policy states, not proof that increasing the cap would improve generation"])
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,cache=ref(root/"cache/result.json"),training={a:ref(root/a/"result.json") for a in ("B3","B2")}))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R12_DIAGNOSE",support,{a:dict(cap=results[a]["cap_active_states"],known_root=results[a]["known_root_rmse_mean_m"]) for a in results},flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

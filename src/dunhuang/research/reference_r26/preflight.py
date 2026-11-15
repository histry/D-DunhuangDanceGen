"""Fail before training on structural, attainable-geometry or FK-gradient violations."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write,sha
from .common import *
from .model import relative_time_bias
from .geometry import fk22,relations,supported_losses

def run(cfg,root,out,replay=None):
    out=fresh(out);forbid_optimizers();pop,data=dataset(root)
    coverage=float(np.mean([r["geometry_support_fraction"] for r in pop["records"] if r["role"]=="train"]))
    assert coverage>=cfg["geometry_support_min_fraction"]
    d,n,s,f,e,info,a=runtime(cfg,"C0")
    p=packet([data[pop["train_probe"][0]]["raw139"][64:160,7:]])
    tests=[];states={}
    for arm in cfg["arms"]:
        torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
        adapter=Adapter(arm!="C0",width=64,part_width=16,layers=(1,4,7),hidden=d.model.latent_dim).cuda()
        states[arm]={k:v.detach().cpu() for k,v in adapter.state_dict().items()}
        install(adapter,p);memory=adapter.context[0].detach().clone()
        swapped={k:v.clone() for k,v in p.items()};swapped["features"][:,:,[16,20]]=swapped["features"][:,:,[20,16]]
        install(adapter,swapped);delta=float((adapter.context[0]-memory).abs().max())
        adapter.double()
        dp={k:v if v.dtype==torch.bool else v.double() for k,v in p.items()}
        install(adapter,dp);before64=adapter.context[0].detach().clone()
        sp={k:v.clone() for k,v in dp.items()};sp["features"][:,:,[16,20]]=sp["features"][:,:,[20,16]]
        install(adapter,sp);delta64=float((adapter.context[0]-before64).abs().max())
        assert delta64<1e-12 if arm=="C0" else delta64>1e-5
        tests.append(dict(arm=arm,shoulder_wrist_swap_memory_max_float32=delta,shoulder_wrist_swap_memory_max_float64=delta64,parameters=sum(v.numel() for v in adapter.parameters())))
    assert states["C1"].keys()==states["C2"].keys()
    assert all(torch.equal(v,states["C2"][k]) for k,v in states["C1"].items())
    common=[k for k in states["C0"] if k in states["C1"]]
    assert all(torch.equal(states["C0"][k],states["C1"][k]) for k in common)
    y=torch.tensor(data[pop["train_probe"][0]]["target"][None,64:160],device="cuda",requires_grad=True)
    pred=relations(fk22(n.unnormalize(y.clone()),s))
    target=pred.detach()+.01
    loss,_=supported_losses(pred,target,torch.ones(pred.shape[:-1],device="cuda",dtype=torch.bool))
    grad=torch.autograd.grad(loss,y)[0];assert torch.isfinite(grad).all() and grad[:,:,7:].norm()>0
    knots=np.array(cfg["synthetic_warp_control_points"]);slopes=np.diff(knots[:,1])/np.diff(knots[:,0])
    assert (slopes>=.75).all() and (slopes<=1.33).all()
    mapping=torch.tensor(np.interp(np.arange(96),knots[:,0],knots[:,1])[None],device="cuda",dtype=torch.float32)
    bias=relative_time_bias(mapping,96,cfg["time_bias_strength"])
    assert (bias[:,:,1:].argmax(-1)-mapping.round()).abs().max()<=1
    result=dict(training_data_ready=True,geometry_support_fraction=coverage,identity=tests,common_initial_tensors_exact=len(common),
        C1_C2_initial_exact=True,geometry_gradient_norm=float(grad.norm()),all_gradients_finite=True,
        synthetic_time_map=dict(control_points=cfg["synthetic_warp_control_points"],slopes=slopes.tolist(),max_shift_frames=float((mapping-torch.arange(96,device="cuda")).abs().max()),attention_only_verified=True,generated_warp_recovery_verified=False),
        C3_training_admitted=False,optimizer_steps=0,operational_qualified=False)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info))
    print("R26_PREFLIGHT_PASSED",tests,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

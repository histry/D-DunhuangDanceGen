"""Train-only architecture choice and one sealed physical-loss scale calibration."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import cohort,runtime,checked_parent
from .objective import objective
from dunhuang.research.time_r29.common import packet
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,digest,readout_check,same_arrays,forbid_optimizers

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    selection=read(root/"reading_train/result.json")
    assert {r["role"] for r in selection["records"]}=={"train"}
    def correct(arm):return [r for r in selection["records"] if r["arm"]==arm and r["mode"]=="correct"]
    def mean(arm,key):return float(np.mean([r["metrics"][key] for r in correct(arm)]))
    source_gate=next(r for r in selection["gates"] if r["arm"]=="R1")
    aligned=source_gate["recordings_passed"]==4 and all(mean("R1",k)<=1.1*mean("R0",k) for k in ("body22_rad","pelvis_wrist_rmse_m"))
    chosen="R1" if aligned else "R0"
    data,ledger,original=cohort(cfg)
    checkpoint=root/"models/R1/adapter.pt" if aligned else checked_parent(cfg,"models/COVER/adapter.pt")
    d,n,s,f,e,info,a=runtime(cfg,aligned,checkpoint);model=digest(a);readouts=[];arrays={}
    for step in cfg["temporal_calibration_steps"]:
        row=ledger[step];v=[data[c] for c in row["cases"]]
        y=torch.tensor(np.stack([x["model_target"] for x in v]),device="cuda")
        m=torch.tensor(np.stack([x["music"] for x in v]),device="cuda")
        target=torch.tensor(np.stack([x["model_points"] for x in v]),device="cuda")
        base,parts,pred,raw,points,r,b=objective(d,a,y,m,packet(v),row["t"],row["seed"],n,s,target,inspect=True)
        with torch.no_grad():official=f(raw.detach(),s)[:,:,:22]
        difference=float((official-points.detach()).abs().max())
        assert difference<=2e-5,"Differentiable FK must match actual official clipped decoder"
        gr=torch.autograd.grad(r,pred,retain_graph=True)[0];gb=torch.autograd.grad(b,pred)[0]
        saturated=pred.detach().abs()>1
        norms=dict(root=float(gr.norm()),body=float(gb.norm()),
            saturated_root_max=float(gr[saturated].abs().max()) if saturated.any() else 0.,
            saturated_body_max=float(gb[saturated].abs().max()) if saturated.any() else 0.)
        assert norms["saturated_root_max"]==norms["saturated_body_max"]==0
        readouts.append(dict(step=step,cases=row["cases"],t=row["t"],seed=row["seed"],body6D=parts["body"],
            Root_physical=float(r.detach()),body_physical=float(b.detach()),FK_max_difference_m=difference,
            target_Core_outside_normalized_support={key:float((y[:,64:160,sl].abs()>1).float().mean()) for key,sl in (("C4",slice(0,4)),("Root",slice(4,7)),("body",slice(7,139)))},
            saturated_Core_fraction={key:float(saturated[:,64:160,sl].float().mean()) for key,sl in (("C4",slice(0,4)),("Root",slice(4,7)),("body",slice(7,139)))},
            gradients=norms))
        arrays[str(step)+"_normalized_prediction"]=pred.detach().cpu().numpy()
        arrays[str(step)+"_actual_clipped_points"]=points.detach().cpu().numpy()
    med=lambda key:float(np.median([r[key] for r in readouts]))
    low,high=cfg["temporal_weight_bounds"]
    weights={name:float(np.clip(cfg["temporal_loss_fraction_per_family"]*med("body6D")/max(med(key),1e-12),low,high))
             for name,key in (("root","Root_physical"),("body","body_physical"))}
    paired_error={name:float(np.mean([z["paired_dynamics"][name]["1"]["rmse"] for z in correct(chosen)])) for name in ("Root_translation","Root_relative_body")}
    admitted=all(np.isfinite(r["gradients"]["root"]) and np.isfinite(r["gradients"]["body"]) and r["gradients"]["root"]>0 and r["gradients"]["body"]>0 for r in readouts) and all(v>1e-4 for v in paired_error.values())
    actual=[];checks=[]
    if replay:
        old=read(Path(replay)/"result.json")
        for x,y in zip(readouts,old["readouts"]):checks.append(readout_check(x,y))
        actual=readouts;readouts=old["readouts"]
    result=dict(chosen_baseline=chosen,aligned=aligned,weights=weights,temporal_training_admitted=bool(admitted),
        rule="Train-only R1 requires4/4 and body/local means<=1.1 R0, else frozen R0; loss family weight=.1 median Train body6D / median physical loss, bounded1e-6..1",
        paired_velocity_error_on_Train=paired_error,readouts=readouts,actual_readouts=actual,replay_checks=checks,
        checkpoint_sha256=sha(checkpoint),reading_Train_evidence_sha256=sha(root/"reading_train/result.json"),
        normalizer_unclipped_substitute_used=False,optimizer_steps=0,operational_qualified=False)
    assert digest(a)==model;a.close();path=out/"arrays.npz";np.savez_compressed(path,**arrays)
    if replay:
        assert {k:v for k,v in result.items() if k not in ("actual_readouts","replay_checks")}=={k:v for k,v in old.items() if k not in ("actual_readouts","replay_checks")}
        same_arrays(path,Path(replay)/path.name)
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),arrays_sha256=sha(path),checkpoint=info))
    print("R30_CALIBRATION",chosen,weights,"ADMITTED",admitted,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)

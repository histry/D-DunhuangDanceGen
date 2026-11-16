"""Same R28 optimization recipe; only the explicit reference clock is enabled."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from .common import runtime,packet,rows as study_rows
from dunhuang.research.contract_r28.train import objective,curves
from dunhuang.research.state_r27.common import read,write,sha,fresh,forbid_optimizers,digest,checkpoint,restore,vector,gradient_norms,readout_check,commit,same_arrays
from dunhuang.research.state_r27.qualification import fitting_gate

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out);assert arm in ("B1","F1")
    if replay:forbid_optimizers()
    domain="aist" if arm=="B1" else "fine";data,cases=study_rows(cfg,root,domain,"train")
    cfg=dict(cfg,train_cases=cases);contract="model"
    d,n,s,f,e,info,a=runtime(cfg,cfg["beta"])
    initial=digest(a);frozen=digest(d.model);y=torch.tensor(np.stack([x[contract+"_target"] for x in data]),device="cuda")
    music=torch.tensor(np.stack([x["music"] for x in data]),device="cuda");ref=packet(data)
    curves_list=[curves(cfg,d,a,n,s,f,data,contract,0,out)];observations=[];history=[];checks=[];actual=[];started=time.monotonic()
    if not replay:
        (out/"snapshots").mkdir();checkpoint(a,out/"initial.pt",cfg,arm,0);opt=torch.optim.AdamW(a.parameters(),lr=cfg["learning_rate"])
        for i in range(cfg["train_steps"]):
            t=cfg["sampler_times"][(i*17)%50];seed=cfg["training_noise_seed"]+i;watch=i in cfg["snapshot_steps"]
            if watch:before=vector(a).clone();checkpoint(a,out/f"snapshots/{i}_pre.pt",cfg,arm,i)
            opt.zero_grad(set_to_none=True);loss,parts=objective(d,a,y,music,ref,t,seed)
            assert torch.isfinite(loss);loss.backward();g=gradient_norms(a) if watch else None
            torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(step=i,t=t,seed=seed,loss=float(loss.detach()),parts=parts,cases=cfg["train_cases"]))
            if watch:
                checkpoint(a,out/f"snapshots/{i}_post.pt",cfg,arm,i+1)
                observations.append(dict(step=i,loss=float(loss.detach()),parts=parts,gradients=g,update_norm=float((vector(a)-before).norm())))
            if i+1 in cfg["curve_updates"]:
                checkpoint(a,out/f"weights_{i+1}.pt",cfg,arm,i+1);curves_list.append(curves(cfg,d,a,n,s,f,data,contract,i+1,out))
            if (i+1)%200==0:print("R29_TRAIN",arm,i+1,float(loss.detach()),flush=True)
        steps=cfg["train_steps"];checkpoint(a,out/"adapter.pt",cfg,arm,steps)
    else:
        old=read(Path(replay)/"result.json");assert initial==old["initial_digest"]
        for name,h in old["files"].items():assert sha(Path(replay)/name)==h
        for item in old["observations"]:
            i=item["step"];row=old["history"][i];restore(a,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(a).clone();a.zero_grad(set_to_none=True)
            loss,parts=objective(d,a,y,music,ref,row["t"],row["seed"]);loss.backward();g=gradient_norms(a)
            restore(a,Path(replay)/f"snapshots/{i}_post.pt")
            current=dict(step=i,loss=float(loss.detach()),parts=parts,gradients=g,update_norm=float((vector(a)-before).norm()))
            checks.append(readout_check(current,item));actual.append(current);write(out/f"actual_{i}.json",current)
        for curve in old["curves"][1:]:
            i=curve["update"];restore(a,Path(replay)/f"weights_{i}.pt");current=curves(cfg,d,a,n,s,f,data,contract,i,out)
            assert current==curve;same_arrays(out/f"curve_{i}.npz",Path(replay)/f"curve_{i}.npz");curves_list.append(current)
        history=old["history"];observations=old["observations"];steps=0;restore(a,Path(replay)/"adapter.pt")
    assert digest(d.model)==frozen and all(p.grad is None for p in d.model.parameters())
    result=dict(arm=arm,contract=contract,initial_digest=initial,final_digest=digest(a),frozen_digest=frozen,
        parameters=sum(p.numel() for p in a.parameters()),optimizer_steps=steps,curves=curves_list,fit=fitting_gate(curves_list[-1]["records"],curves_list[0]["records"]),
        history=history,observations=observations,actual_readouts=actual,replay_checks=checks,per_target_presentations=1200,
        files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        reference_time_beta=cfg["beta"],Core_C4_loss_applied=False,full_139_input=True,reference_representation="CANONICAL_SO3_ROWS",wall_seconds=time.monotonic()-started,operational_qualified=False)
    if replay:assert result["final_digest"]==old["final_digest"]
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info))
    a.close();print("R29_TRAIN_DONE",arm,steps,result["fit"]["passed"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",required=True,choices=["B1","F1"]);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.root,args.out,args.arm,args.replay_from)

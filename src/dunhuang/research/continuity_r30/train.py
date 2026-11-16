"""Two bounded matched-budget arms; original COVER history defines every training batch."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import cohort,runtime,parent
from .objective import objective
from dunhuang.research.time_r29.common import packet
from dunhuang.research.contract_r28.train import curves
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,digest,checkpoint,restore,vector,gradient_norms,readout_check,same_arrays,forbid_optimizers

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    audit=read(root/"audit/result.json")
    weights=None
    if arm=="R1":
        assert audit["reading_control_admitted"];aligned=True
    else:
        decision=read(root/"calibration/result.json");assert decision["temporal_training_admitted"]
        aligned=decision["aligned"];weights=decision["weights"]
    data,ledger,original=cohort(cfg)
    d,n,s,f,e,info,a=runtime(cfg,aligned);initial=digest(a);frozen=digest(d.model)
    assert initial==original["initial_digest"] and frozen==original["frozen_digest"]
    probes=[data[c] for c in cfg["train_cases"]];oldcfg=read("src/dunhuang/research/time_r29/config.json")
    cs=[curves(oldcfg,d,a,n,s,f,probes,"model",0,out)];history=[];obs=[];actual=[];checks=[]
    def loss(row):
        v=[data[c] for c in row["cases"]]
        y=torch.tensor(np.stack([x["model_target"] for x in v]),device="cuda")
        m=torch.tensor(np.stack([x["music"] for x in v]),device="cuda")
        target=torch.tensor(np.stack([x["model_points"] for x in v]),device="cuda")
        return objective(d,a,y,m,packet(v),row["t"],row["seed"],n,s,target,weights)
    if not replay:
        (out/"snapshots").mkdir();checkpoint(a,out/"initial.pt",cfg,arm,0);opt=torch.optim.AdamW(a.parameters(),lr=cfg["learning_rate"])
        for row in ledger:
            i=row["step"];watch=i in cfg["snapshot_steps"]
            if watch:before=vector(a).clone();checkpoint(a,out/f"snapshots/{i}_pre.pt",cfg,arm,i)
            opt.zero_grad(set_to_none=True);value,parts=loss(row);assert torch.isfinite(value);value.backward()
            g=gradient_norms(a) if watch else None
            torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(row,loss=float(value.detach()),parts=parts))
            if watch:
                checkpoint(a,out/f"snapshots/{i}_post.pt",cfg,arm,i+1);obs.append(dict(step=i,loss=float(value.detach()),parts=parts,gradients=g,update_norm=float((vector(a)-before).norm())))
            if i+1 in cfg["curve_updates"]:
                checkpoint(a,out/f"weights_{i+1}.pt",cfg,arm,i+1);cs.append(curves(oldcfg,d,a,n,s,f,probes,"model",i+1,out))
            if (i+1)%200==0:print("R30_TRAIN",arm,i+1,float(value.detach()),flush=True)
        checkpoint(a,out/"adapter.pt",cfg,arm,1200);steps=1200
    else:
        old=read(Path(replay)/"result.json");assert initial==old["initial_digest"]
        for name,h in old["files"].items():assert sha(Path(replay)/name)==h
        for item in old["observations"]:
            i=item["step"];restore(a,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(a).clone();a.zero_grad(set_to_none=True)
            value,parts=loss(ledger[i]);value.backward();g=gradient_norms(a);restore(a,Path(replay)/f"snapshots/{i}_post.pt")
            current=dict(step=i,loss=float(value.detach()),parts=parts,gradients=g,update_norm=float((vector(a)-before).norm()))
            checks.append(readout_check(current,item));actual.append(current);write(out/f"actual_{i}.json",current)
        for oldcurve in old["curves"][1:]:
            step=oldcurve["update"];restore(a,Path(replay)/f"weights_{step}.pt");c=curves(oldcfg,d,a,n,s,f,probes,"model",step,out)
            assert c==oldcurve;cs.append(c);same_arrays(out/f"curve_{step}.npz",Path(replay)/f"curve_{step}.npz")
        restore(a,Path(replay)/"adapter.pt");history=old["history"];obs=old["observations"];steps=0
    assert digest(d.model)==frozen and all(p.grad is None for p in d.model.parameters())
    result=dict(arm=arm,aligned=aligned,weights=weights,parameters=sum(p.numel() for p in a.parameters()),optimizer_steps=steps,
        initial_digest=initial,final_digest=digest(a),frozen_digest=frozen,history=history,curves=cs,observations=obs,
        actual_readouts=actual,replay_checks=checks,per_source_presentations=300,recordings=16,music_groups=8,
        files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        full_native_status="NOT_EVALUABLE",operational_qualified=False)
    if replay:assert result["final_digest"]==old["final_digest"]
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,COVER_history_sha256=sha(parent(cfg)/"models/COVER/result.json")))
    a.close();print("R30_TRAIN_DONE",arm,steps,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["R1","R2"],required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.arm,a.replay_from)

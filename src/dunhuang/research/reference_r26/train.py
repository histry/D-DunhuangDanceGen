"""Fixed matched C0-C2 training; actual saved pre/post update replay without optimizers."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import sha,write
from .common import *

def run(cfg,root,out,arm,replay=None):
    out=fresh(out)
    if replay:forbid_optimizers()
    d,n,s,f,e,info,adapter=runtime(cfg,arm);adapter.attach(d.model)
    pop,data=dataset(root);rows=ledger(cfg,pop["train_cases"]);frozen=digest(d.model);initial=digest(adapter)
    curves=[];observations=[];replayed=[];checks=[];history=[];started=time.monotonic()
    def curve(update):
        scores=[];arrays={}
        with torch.no_grad():
            for case in pop["train_probe"][:2]:
                row=dict(cases=[case,case],seed=cfg["seed"]+5000,reference_drop=[False,False],occluded=[False,False])
                y,m,p=batch(cfg,data,row)
                loss,pred,parts,_=objective(d,adapter,y,m,p,519,row["seed"],cfg,arm,n,s)
                scores.append(dict(case=case,loss=float(loss),parts=parts));arrays[str(case)]=pred.cpu().numpy()
        path=out/f"curve_{update}.npz";np.savez_compressed(path,**arrays)
        if replay:same_arrays(path,Path(replay)/path.name)
        curves.append(dict(update=update,train_only=True,scores=scores))
    curve(0)
    if not replay:
        (out/"snapshots").mkdir();checkpoint(adapter,out/"initial.pt",cfg,arm,0)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg["learning_rate"])
        for row in rows:
            i=row["step"];watch=i in cfg["snapshot_steps"];y,m,p=batch(cfg,data,row)
            before=vector(adapter) if watch else None
            if watch:checkpoint(adapter,out/f"snapshots/{i}_pre.pt",cfg,arm,i)
            optimizer.zero_grad(set_to_none=True)
            loss,pred,parts,layers=objective(d,adapter,y,m,p,row["t"],row["seed"],cfg,arm,n,s,watch)
            assert torch.isfinite(loss) and loss.requires_grad;loss.backward()
            grads=gradients(adapter) if watch else None
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            history.append(dict(step=i,loss=float(loss.detach()),parts=parts))
            if watch:
                checkpoint(adapter,out/f"snapshots/{i}_post.pt",cfg,arm,i+1)
                observations.append(dict(step=i,loss=float(loss.detach()),parts=parts,layers=layers,gradient_norms=grads,update_norm=float((vector(adapter)-before).norm())))
            if i+1 in cfg["curve_updates"]:curve(i+1)
            if (i+1)%200==0:print("R26_TRAIN",arm,i+1,history[-1],flush=True)
        checkpoint(adapter,out/"adapter.pt",cfg,arm,len(rows));steps=len(rows)
    else:
        old=read(Path(replay)/"result.json");assert old["initial_digest"]==initial
        for name,h in old["checkpoint_hashes"].items():assert sha(Path(replay)/name)==h
        for expected in old["observations"]:
            i=expected["step"];row=rows[i];y,m,p=batch(cfg,data,row)
            restore(adapter,Path(replay)/f"snapshots/{i}_pre.pt",cfg,arm);before=vector(adapter);adapter.zero_grad(set_to_none=True)
            loss,pred,parts,layers=objective(d,adapter,y,m,p,row["t"],row["seed"],cfg,arm,n,s,True);loss.backward();g=gradients(adapter)
            restore(adapter,Path(replay)/f"snapshots/{i}_post.pt",cfg,arm)
            actual=dict(step=i,loss=float(loss.detach()),parts=parts,layers=layers,gradient_norms=g,update_norm=float((vector(adapter)-before).norm()))
            write(out/f"actual_{i}.json",actual);checks.append(check_readout(actual,expected));replayed.append(actual)
        for update in cfg["curve_updates"][1:]:
            restore(adapter,Path(replay)/("snapshots/599_post.pt" if update==600 else "adapter.pt"),cfg,arm);curve(update)
        assert curves==old["curves"];history=old["history"];observations=old["observations"];steps=0
        restore(adapter,Path(replay)/"adapter.pt",cfg,arm)
    assert digest(d.model)==frozen and all(p.grad is None for p in d.model.parameters())
    result=dict(arm=arm,initial_digest=initial,final_digest=digest(adapter),frozen_digest=frozen,parameters=sum(p.numel() for p in adapter.parameters()),
        curves=curves,observations=observations,actual_replay_readouts=replayed,replay_checks=checks,history=history,ledger=rows,
        checkpoint_hashes=old["checkpoint_hashes"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*.pt")},
        optimizer_steps=steps,wall_seconds=time.monotonic()-started,operational_qualified=False,whole_song_generations=0)
    if replay:assert result["final_digest"]==old["final_digest"] and rows==old["ledger"]
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info,selection="Fixed final1200, no DEV selection"))
    adapter.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["C0","C1","C2"],required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.arm,a.replay_from)

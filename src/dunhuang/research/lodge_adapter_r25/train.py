"""Two matched real optimizations; replay uses saved actual pre/post values, never an optimizer."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write,sha
from .common import runtime,dataset,ledger,batch,objective,checkpoint,restore,vector,read,fresh,commit,forbid_optimizers,same_arrays
from .model import digest,grad_readout

def run(cfg,root,out,arm,replay=None):
    out=fresh(out)
    if replay:forbid_optimizers()
    d,n,s,f,e,info,adapter=runtime(cfg);adapter.phase=arm=="phase";adapter.attach(d.model)
    pop,data=dataset(root);rows=ledger(cfg,pop["train_cases"]);frozen=digest(d.model);initial=digest(adapter)
    steps=[];curves=[];observations=[];started=time.monotonic()
    if not replay:(out/"snapshots").mkdir();checkpoint(adapter,out/"initial.pt",cfg,arm,0)
    def curve(update):
        values={};scores=[]
        with torch.no_grad():
            for case in [pop["train_cases"][0],pop["train_cases"][2]]:
                row=dict(cases=[case,case],seed=cfg["seed"]+5000,reference_drop=[False,False])
                y,m,p=batch(cfg,data,row)
                loss,pred,parts,_=objective(d,adapter,y,m,p,519,row["seed"],cfg)
                scores.append(dict(case=case,loss=float(loss),parts=parts));values[str(case)]=pred.cpu().numpy()
        path=out/f"curve_{update}.npz";np.savez_compressed(path,**values)
        if replay:same_arrays(path,Path(replay)/path.name)
        curves.append(dict(update=update,train_only=True,scores=scores))
    curve(0)
    if not replay:
        opt=torch.optim.AdamW(adapter.parameters(),lr=cfg["learning_rate"])
        for row in rows:
            i=row["step"];y,m,p=batch(cfg,data,row);watch=i in cfg["snapshot_steps"];before=vector(adapter) if watch else None
            if watch:checkpoint(adapter,out/f"snapshots/{i}_pre.pt",cfg,arm,i)
            opt.zero_grad(set_to_none=True);loss,pred,parts,layers=objective(d,adapter,y,m,p,row["t"],row["seed"],cfg,watch)
            assert torch.isfinite(loss) and loss.requires_grad;loss.backward()
            grads=grad_readout(adapter) if watch else None;torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);opt.step()
            steps.append(dict(step=i,loss=float(loss.detach()),parts=parts))
            if watch:
                checkpoint(adapter,out/f"snapshots/{i}_post.pt",cfg,arm,i+1)
                observations.append(dict(step=i,before_loss=float(loss.detach()),parts=parts,gradient_norms=grads,layers=layers,
                    update_norm=float((vector(adapter)-before).norm())))
            if i+1 in cfg["curve_updates"]:curve(i+1)
            if (i+1)%100==0:print("R25_TRAIN",arm,i+1,steps[-1],flush=True)
        checkpoint(adapter,out/"adapter.pt",cfg,arm,len(rows));optimizer_steps=len(rows)
    else:
        old=read(Path(replay)/"result.json");assert old["initial_digest"]==initial
        for name,h in old["checkpoint_hashes"].items():assert sha(Path(replay)/name)==h
        for item in old["observations"]:
            i=item["step"];row=rows[i];y,m,p=batch(cfg,data,row);restore(adapter,Path(replay)/f"snapshots/{i}_pre.pt",cfg,arm);before=vector(adapter)
            adapter.zero_grad(set_to_none=True);loss,pred,parts,layers=objective(d,adapter,y,m,p,row["t"],row["seed"],cfg,True);loss.backward();grads=grad_readout(adapter)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);restore(adapter,Path(replay)/f"snapshots/{i}_post.pt",cfg,arm)
            actual=dict(step=i,before_loss=float(loss.detach()),parts=parts,gradient_norms=grads,layers=layers,update_norm=float((vector(adapter)-before).norm()))
            assert actual==item,("Actual update replay",arm,i)
        # Curves after0 correspond to immutable saved checkpoints.
        for update in cfg["curve_updates"][1:]:
            restore(adapter,Path(replay)/("snapshots/299_post.pt" if update==300 else "adapter.pt"),cfg,arm);curve(update)
        assert curves==old["curves"];observations=old["observations"];steps=old["history"];optimizer_steps=0
        restore(adapter,Path(replay)/"adapter.pt",cfg,arm)
    assert digest(d.model)==frozen and not any(p.grad is not None for p in d.model.parameters())
    result=dict(arm=arm,initial_digest=initial,final_digest=digest(adapter),frozen_digest=frozen,parameters=sum(p.numel() for p in adapter.parameters()),
        curves=curves,observations=observations,history=steps,ledger=rows,checkpoint_hashes=old["checkpoint_hashes"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*.pt")},optimizer_steps=optimizer_steps,wall_seconds=time.monotonic()-started,
        operational_qualified=False,whole_song_generations=0)
    if replay:assert result["final_digest"]==old["final_digest"] and rows==old["ledger"]
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,data=str(Path(root)/"data/result.json"),selection="Fixed final step; no DEV selection"))
    adapter.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["event","phase"],required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.arm,a.replay_from)

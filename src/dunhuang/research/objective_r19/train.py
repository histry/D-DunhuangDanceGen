"""Matched single-task ablation with saved actual AdamW pre/post states."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.masked_denoiser.common import tensor_hash
from dunhuang.research.global_clock_r17.train import digest
from .common import load,restore,capture,cache_data,state,parameters,vector,schedule
from .diagnostics import build,analyze,backward,after_update
from .probes import run as probes

def run(cfg,calibration,out,arm,replay=None):
    out=fresh(out);calibration_ref=ref(Path(calibration)/"result.json");weights=read(checked(calibration_ref))["weights"]
    if replay:forbid_optimizers()
    mode="unique" if arm=="U" else "self"
    w,d,data,predict,head=load(cfg,mode);cache=cache_data(cfg,w,d,data)
    frozen=w.digest(frozen=True);initial_motion=w.digest();initial_head=digest(head)
    initial_q,initial_scores=probes(cfg,cache,predict,head)
    np.savez_compressed(out/"initial_q.npz",**initial_q)
    ledger=schedule(cfg);history=[];observations=[]
    if not replay:
        opt=torch.optim.AdamW(parameters(w,head),lr=cfg["learning_rate"])
        (out/"snapshots").mkdir();updates=0
        for row in ledger:
            step=row["step"];s=state(cfg,cache,row["case"],row["t"],row["hard"],row["seed"])
            y,m,o,v,k,h,meta,t,xt=s
            pred,terms,counts,active,objective=build(cfg,predict,cache,s,arm,weights,step%2==0)
            if not torch.isfinite(objective):raise ValueError("Nonfinite registered loss")
            watch=step in cfg["snapshots"]
            if watch:
                pre=capture(cfg,w,head,out/"snapshots"/f"{step}_pre.pt",arm,step)
                before=vector(w,head);record,gu=analyze(cfg,w,head,predict,cache,s,pred,terms,counts,active)
            clipped=backward(w,head,objective);opt.step();updates+=1
            if watch:
                delta=vector(w,head)-before
                result=after_update(record,gu,delta,predict,s)
                post=capture(cfg,w,head,out/"snapshots"/f"{step}_post.pt",arm,step+1)
                observations.append(dict(step=step,pre=pre,post=post,readout=dict(result,clipping=clipped)))
            history.append(dict(step=step,loss=float(objective.detach()),clipping=clipped,
                unknown=float(terms["unknown"].detach()),observed=float(terms["observed"].detach())))
            row.update(x_t_sha256=tensor_hash(xt),known_sha256=tensor_hash(k))
            if (step+1)%40==0:print("R19_TRAIN",arm,step+1,history[-1]["unknown"],flush=True)
        checkpoint=observations[-1]["post"];assert observations[-1]["step"]==cfg["steps"]-1
    else:
        old=read(Path(replay)/"result.json");assert initial_motion==old["initial_motion"] and initial_head==old["initial_head"]
        assert initial_scores==old["initial_scores"];same_arrays(out/"initial_q.npz",Path(replay)/"initial_q.npz")
        for item in old["update_observations"]:
            row=ledger[item["step"]];s=state(cfg,cache,row["case"],row["t"],row["hard"],row["seed"])
            restore(cfg,w,head,item["pre"])
            pred,terms,counts,active,objective=build(cfg,predict,cache,s,arm,weights,item["step"]%2==0)
            before=vector(w,head);record,gu=analyze(cfg,w,head,predict,cache,s,pred,terms,counts,active)
            clipped=backward(w,head,objective)
            restore(cfg,w,head,item["post"]);delta=vector(w,head)-before
            result=after_update(record,gu,delta,predict,s)
            assert dict(result,clipping=clipped)==item["readout"],"Saved actual-update replay differs"
            observations.append(item)
        checkpoint=old["checkpoint"];restore(cfg,w,head,checkpoint);updates=0
        for row in ledger:
            s=state(cfg,cache,row["case"],row["t"],row["hard"],row["seed"])
            row.update(x_t_sha256=tensor_hash(s[-1]),known_sha256=tensor_hash(s[4]))
        assert ledger==old["ledger"]
    q,scores=probes(cfg,cache,predict,head);np.savez_compressed(out/"q_eval.npz",**q)
    assert w.digest(frozen=True)==frozen
    result=dict(arm=arm,mode=mode,checkpoint=checkpoint,optimizer_steps=updates,
        initial_motion=initial_motion,initial_head=initial_head,final_motion=w.digest(),final_head=digest(head),
        frozen_digest=frozen,initial_scores=initial_scores,scores=scores,update_observations=observations,
        history=history,ledger=ledger,calibration=calibration_ref,fk_parity_max_m=cache["parity"],
        motion_parameters=sum(p.numel() for p in w.selected()),head_parameters=sum(p.numel() for p in head.parameters()),
        whole_song_generations=0,operational_qualified=False)
    if replay:
        assert scores==old["scores"] and result["final_motion"]==old["final_motion"] and result["final_head"]==old["final_head"]
        same_arrays(out/"q_eval.npz",Path(replay)/"q_eval.npz")
    checked(calibration_ref)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        calibration=ref(Path(calibration)/"protocol.json"),task="inbetween",fit_cases=cfg["cases"],
        validation_selection=False,actual_updates_observed=cfg["snapshots"],replay_optimizer_steps=0))
    predict.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--calibration",required=True);p.add_argument("--out",required=True)
    p.add_argument("--arm",choices=["N0","N1","N2","U"],required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.calibration,a.out,a.arm,a.replay_from)

"""Tiny learnability first, then gated population adaptation; replay never trains."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.masked_denoiser.common import tensor_hash
from dunhuang.research.global_clock_r17.engine import tensors,observed
from dunhuang.research.global_clock_r17.contract import metadata
from dunhuang.research.global_clock_r17.model import fk
from dunhuang.research.global_clock_r17.objective import losses
from dunhuang.research.global_clock_r17.train import digest
from .engine import load,population,unknown_mse,trainable,supervision,amplitude

def probes(cfg,rows,cache,predict,head):
    arrays={};scores=[]
    with torch.no_grad():
        for row in rows:
            z,y,m,o=cache[row["role"],row["case"]]
            for task in cfg["tasks"]:
                v,k,h=observed(y,task,prefix=y[0,:75].cpu().numpy() if task=="progressive" else None)
                meta=torch.tensor(metadata()[None],device=y.device)
                for timestep in (519,19):
                    seed=cfg["seed"]+9000+row["case"]*100+timestep+(10000 if row["role"]=="eval" else 0)
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    t=torch.tensor([timestep],device=y.device);noise=torch.randn_like(y)
                    # Explicit shared noise, independent of CF order.
                    diffusion=cache["diffusion"];xt=diffusion.q_sample(y,t,noise=noise)
                    for control in cfg["controls"]:
                        p,_=predict(xt,m,v,k,meta,t,control)
                        name=f'{row["role"]}_{row["case"]}_{task}_{timestep}_{control}'
                        arrays[name]=p[0].cpu().numpy()
                        scores.append(dict(role=row["role"],case=row["case"],task=task,t=timestep,
                            control=control,unknown_mse=float(unknown_mse(p,y,k)),attention=dict(head.last)))
                    correct=arrays[f'{row["role"]}_{row["case"]}_{task}_{timestep}_correct']
                    perm=arrays[f'{row["role"]}_{row["case"]}_{task}_{timestep}_key_order']
                    assert np.max(abs(correct-perm))<1e-5,"Key ordering changed the correspondence"
    return arrays,scores

def run(cfg,out,mode,stage,replay=None):
    out=fresh(out);pre,fit,ev=population(cfg)
    fit=[r for r in fit if r["case"] in cfg["tiny_cases"]] if stage=="tiny" else fit
    rows=fit if stage=="tiny" else fit+ev
    steps=cfg["tiny_steps"] if stage=="tiny" else cfg["main_steps"]
    old=read(Path(replay)/"result.json") if replay else None
    if replay:forbid_optimizers()
    w,d,data,predict,head=load(cfg,mode,old["checkpoint"] if replay else None)
    frozen=w.digest(frozen=True);initial_motion=old["initial_motion"] if replay else w.digest()
    initial_head=old["initial_head"] if replay else digest(head)
    parity=np.load(checked(pre["parity"]));parents=list(map(int,parity["parents"]))
    offsets=torch.tensor(parity["offsets"],device="cuda");scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    cache={"diffusion":d};errors=[]
    for row in rows:
        z,y,m=tensors(row);origin=torch.tensor(z["origin"][None],device="cuda")
        with torch.no_grad():error=float(abs(fk(y,scale,bias,origin,parents,offsets)[0].cpu().numpy()-parity[row["role"]+"_"+str(row["case"])]).max())
        assert error<=1e-5;errors.append(error);cache[row["role"],row["case"]]=(z,y,m,origin)
    initial_scores=old["initial_scores"] if replay else None
    if not replay:
        _,initial_scores=probes(cfg,fit,cache,predict,head)
        opt=torch.optim.AdamW(trainable(w,head),lr=cfg["learning_rate"]);rng=np.random.default_rng(cfg["seed"])
    ledger=[];history=[];updates=0
    if not replay:
        for step in range(steps):
            row=fit[int(rng.integers(len(fit)))];z,y,m,origin=cache["fit",row["case"]]
            task,ranking=supervision(step);boundary=int(rng.integers(80,145))
            v,k,h=observed(y,task,hard=(step//6)%2==0,prefix=y[0,:75].cpu().numpy() if task=="progressive" else None,boundary=boundary)
            meta=torch.tensor(metadata(boundary=boundary)[None],device="cuda")
            seed=cfg["seed"]+20000+step;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            t=torch.tensor([int(rng.integers(1000))],device="cuda");xt=d.q_sample(y,t)
            pred,_=predict(xt,m,v,k,meta,t)
            parts=losses(pred,y,v,k,h,scale,bias,origin,parents,offsets)
            parts["amplitude"]=amplitude(pred,y,scale,bias,origin,parents,offsets)
            context=pred.sum()*0
            if ranking:
                wrong,_=predict(xt,m,v,k,meta,t,"content")
                context=torch.relu(cfg["context_margin"]+unknown_mse(pred,y,k)-unknown_mse(wrong,y,k))
            parts["context"]=context;objective=sum(cfg["loss_weights"][name]*value for name,value in parts.items())
            if not torch.isfinite(objective):raise ValueError("Nonfinite objective")
            opt.zero_grad();objective.backward()
            # Task-specific gradients must reach selected motion layers, not just a small head.
            motion_norm=float(torch.nn.utils.clip_grad_norm_(w.selected(),1.,error_if_nonfinite=True))
            head_norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True))
            assert all(p.grad is None for p in w.parameters() if not p.requires_grad)
            opt.step();updates+=1
            history.append(dict(step=step,task=task,ranking=ranking,loss=float(objective.detach()),
                motion_gradient_norm=motion_norm,head_gradient_norm=head_norm,parts={n:float(v.detach()) for n,v in parts.items()}))
            ledger.append(dict(step=step,case=row["case"],group=row["left"]["source_group"],task=task,ranking=ranking,
                t=int(t[0]),boundary=boundary,hard=bool(h.any()),x_t_sha256=tensor_hash(xt),known_sha256=tensor_hash(k)))
            if (step+1)%30==0:print("R18_TRAIN",stage,mode,step+1,history[-1]["loss"],flush=True)
    q,scores=probes(cfg,rows,cache,predict,head);np.savez_compressed(out/"q_eval.npz",**q)
    assert w.digest(frozen=True)==frozen
    task_counts={}
    for task in cfg["tasks"]:
        items=[r for r in history if r["task"]==task]
        task_counts[task]=dict(updates=len(items),ranking_updates=sum(r["ranking"] for r in items),
            motion_nonzero_gradients=sum(r["motion_gradient_norm"]>0 for r in items),
            head_nonzero_gradients=sum(r["head_gradient_norm"]>0 for r in items))
    if replay:
        assert scores==old["scores"] and w.digest()==old["final_motion"] and digest(head)==old["final_head"]
        same_arrays(out/"q_eval.npz",Path(replay)/"q_eval.npz");checkpoint=old["checkpoint"]
    else:
        path=out/"model.pt"
        torch.save(dict(schema=cfg["schema"],config=cfg,mode=mode,motion=w.state(),head=head.state_dict(),
            code_commit=commit(),operational_qualified=False),path);checkpoint=ref(path)
    result=dict(stage=stage,mode=mode,checkpoint=checkpoint,initial_motion=initial_motion,initial_head=initial_head,
        final_motion=w.digest(),final_head=digest(head),frozen_digest=frozen,
        motion_parameters=sum(p.numel() for p in w.selected()),head_parameters=sum(p.numel() for p in head.parameters()),
        optimizer_steps=updates,initial_scores=initial_scores,scores=scores,history=history,ledger=ledger,
        task_counts=task_counts,fk_parity_max_m=max(errors),operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        data=ref(Path(cfg["parent"])/"data/protocol.json"),fit=[r["left"] for r in fit],
        eval=[] if stage=="tiny" else [r["left"] for r in ev],validation_selection=False,
        motion_trainable_names=w.trainable_names,progressive_training="Real past teacher forcing; not generated-history training"))
    predict.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True)
    p.add_argument("--mode",choices=["self","unique"],required=True);p.add_argument("--stage",choices=["tiny","main"],required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.mode,a.stage,a.replay_from)

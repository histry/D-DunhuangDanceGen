"""Finite Train-only positive-control calibration and conditional matched interventions."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .trace import sample,trace_arrays,row_noise
from .qualification import fitting_gate

def dataset(root):
    root=Path(root);protocol=read(root/"teachers/protocol.json")
    for name,h in protocol["files"].items():assert sha(root/"teachers"/name)==h
    z=dict(np.load(root/"teachers/teacher.npz"))
    return z

def objective(d,adapter,n,s,y,music,ref,t,seed,xt=None):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);ts=torch.full((4,),t,device="cuda",dtype=torch.long)
    epsilon=torch.randn_like(y)
    if xt is None:xt=d.q_sample(y,ts,epsilon)
    adapter.clear();rng=torch.cuda.get_rng_state();cpu=torch.get_rng_state()
    with torch.no_grad():base=d.model(xt,music,None,ts,cond_drop_prob=.2)
    torch.cuda.set_rng_state(rng);torch.set_rng_state(cpu);adapter.set_reference(**ref)
    pred=d.model(xt,music,None,ts,cond_drop_prob=.2);adapter.clear()
    parts=dict(body=(pred[:,64:160,7:]-y[:,64:160,7:]).square().mean(),
        model_C4_Root=(pred[:,64:160,:7]-y[:,64:160,:7]).square().mean(),
        velocity=(pred[:,65:160,7:]-pred[:,64:159,7:]-y[:,65:160,7:]+y[:,64:159,7:]).square().mean(),
        outside=torch.cat((pred[:,:64]-base[:,:64],pred[:,160:]-base[:,160:]),1).square().mean())
    loss=parts["body"]+.1*parts["model_C4_Root"]+.1*parts["velocity"]+.25*parts["outside"]
    return loss,{k:float(v.detach()) for k,v in parts.items()}

def curves(cfg,d,a,n,s,f,z,step,out):
    target=torch.tensor(z["target"],device="cuda");music=torch.tensor(z["music"],device="cuda")
    refs={mode:packet(list(z["raw139"][[1,0,3,2] if mode=="wrong" else [0,1,2,3],64:160,7:])) for mode in ("correct","wrong")}
    records=[];arrays={}
    for t in cfg["curve_times"]:
        ts=torch.full((4,),t,device="cuda",dtype=torch.long)
        with row_noise(cfg["curve_noise_seeds"],device="cuda"):
            epsilon=torch.randn_like(target)
        xt=d.q_sample(target,ts,epsilon)
        for mode in ("correct","wrong","none"):
            a.clear()
            if mode!="none":a.set_reference(**refs[mode])
            with torch.no_grad():pred=d.model(xt,music,None,ts,cond_drop_prob=0)
            m=decoded_metrics(pred,target,n,s,f)
            for i in range(4):
                records.append(dict(teacher=i,t=t,mode=mode,
                    raw_mse=float((pred[i,64:160,7:]-target[i,64:160,7:]).square().mean()),
                    clipped_mse=float((pred[i,64:160,7:].clamp(-1,1)-target[i,64:160,7:]).square().mean()),metrics=m[i]))
            arrays[f"{t}_{mode}"]=pred.cpu().numpy()
    a.clear();np.savez_compressed(out/f"curve_{step}.npz",**arrays)
    return dict(update=step,records=records)

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    capacity=arm=="P2";start=root/"models/P/adapter.pt" if arm in ("Q0","Q1") else None
    d,n,s,f,e,info,a=runtime(cfg,capacity,start);z=dataset(root)
    y=torch.tensor(z["target"],device="cuda");music=torch.tensor(z["music"],device="cuda")
    ref=packet(list(z["raw139"][:,64:160,7:]))
    initial=digest(a);frozen=digest(d.model);observations=[];actual=[];checks=[];history=[];curve_rows=[];caches=[]
    initial_curve=curves(cfg,d,a,n,s,f,z,0,out);curve_rows.append(initial_curve)
    budget=cfg["positive_max_steps"] if arm=="P" else read(root/"models/P/result.json")["optimizer_steps"] if arm=="P2" else cfg["continuation_steps"]
    steps_to_watch=cfg["snapshot_steps"];started=time.monotonic()
    cache=None;cache_name=None
    if not replay:
        (out/"snapshots").mkdir();checkpoint(a,out/"initial.pt",cfg,arm,0)
        opt=torch.optim.AdamW(a.parameters(),lr=cfg["learning_rate"])
        for i in range(budget):
            # Both continuation arms pay the same frozen refresh budget; Q0 caches remain shadow only.
            if arm in ("Q0","Q1") and i in cfg["cache_refresh_steps"]:
                cache_name=f"cache_{i}.npz";state=digest(a)
                seeds=[cfg["cache_seed"]+i+j//2 for j in range(4)]
                value,noise,trace=sample(d,a,ref,music,seeds,trace=True,selected=cfg["cache_times"],target=y)
                raw=trace_arrays(trace);indices=[raw["times"].tolist().index(t) for t in cfg["cache_times"]]
                cache=raw["xt"][indices];np.savez_compressed(out/cache_name,xt=cache,times=np.array(cfg["cache_times"]))
                caches.append(dict(step=i,file=cache_name,sha256=sha(out/cache_name),model_digest=state,noise=noise,state_kind="ACTUAL_REVERSE_XT",used_for_loss=arm=="Q1"))
            t=cfg["cache_times"][i%len(cfg["cache_times"])] if arm in ("Q0","Q1") else cfg["sampler_times"][(i*17)%len(cfg["sampler_times"])]
            seed=cfg["training_noise_seed"]+i;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            xt=None;origins=["FORWARD_Q_PROXY"]*4
            if arm=="Q1":
                ts=torch.full((4,),t,device="cuda",dtype=torch.long);xt=d.q_sample(y,ts,torch.randn_like(y))
                state=torch.tensor(cache[cfg["cache_times"].index(t)],device="cuda")
                rows=torch.tensor([(i+j)%2==0 for j in range(4)],device="cuda")
                xt=torch.where(rows[:,None,None],state,xt);origins=["ACTUAL_REVERSE_XT" if b else "FORWARD_Q_PROXY" for b in rows.tolist()]
            watch=i in steps_to_watch;before=vector(a).clone() if watch else None
            if watch:checkpoint(a,out/f"snapshots/{i}_pre.pt",cfg,arm,i)
            opt.zero_grad(set_to_none=True);loss,parts=objective(d,a,n,s,y,music,ref,t,seed,xt)
            assert torch.isfinite(loss);loss.backward();g=gradient_norms(a) if watch else None
            torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(step=i,t=t,seed=seed,loss=float(loss.detach()),parts=parts,input_state_origins=origins,cache=cache_name))
            if watch:
                checkpoint(a,out/f"snapshots/{i}_post.pt",cfg,arm,i+1)
                observations.append(dict(step=i,loss=float(loss.detach()),parts=parts,gradients=g,update_norm=float((vector(a)-before).norm())))
            update=i+1
            if update in cfg["curve_updates"] or update==budget:
                checkpoint(a,out/f"weights_{update}.pt",cfg,arm,update)
                curve=curves(cfg,d,a,n,s,f,z,update,out);curve_rows.append(curve)
                gate=fitting_gate(curve["records"],initial_curve["records"])
                if arm=="P" and update>=cfg["positive_min_steps"] and gate["passed"]:
                    print("R27_TRAIN_FIT_EARLY_FIXED_RULE",update,flush=True);break
            if update%200==0:print("R27_TRAIN",arm,update,history[-1]["loss"],flush=True)
        steps=len(history);checkpoint(a,out/"adapter.pt",cfg,arm,steps)
    else:
        old=read(Path(replay)/"result.json");assert initial==old["initial_digest"]
        for name,h in old["files"].items():assert sha(Path(replay)/name)==h
        for item in old["observations"]:
            i=item["step"];row=old["history"][i];t=row["t"];seed=row["seed"];xt=None
            if arm=="Q1":
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);ts=torch.full((4,),t,device="cuda",dtype=torch.long)
                xt=d.q_sample(y,ts,torch.randn_like(y));cache=dict(np.load(Path(replay)/row["cache"]))
                state=torch.tensor(cache["xt"][cache["times"].tolist().index(t)],device="cuda");rows=torch.tensor([x=="ACTUAL_REVERSE_XT" for x in row["input_state_origins"]],device="cuda")
                xt=torch.where(rows[:,None,None],state,xt)
            restore(a,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(a).clone();a.zero_grad(set_to_none=True)
            loss,parts=objective(d,a,n,s,y,music,ref,t,seed,xt);loss.backward();g=gradient_norms(a)
            restore(a,Path(replay)/f"snapshots/{i}_post.pt")
            row=dict(step=i,loss=float(loss.detach()),parts=parts,gradients=g,update_norm=float((vector(a)-before).norm()))
            write(out/f"actual_{i}.json",row);checks.append(readout_check(row,item));actual.append(row)
        for curve in old["curves"][1:]:
            update=curve["update"];restore(a,Path(replay)/f"weights_{update}.pt")
            c=curves(cfg,d,a,n,s,f,z,update,out);assert c==curve;same_arrays(out/f"curve_{update}.npz",Path(replay)/f"curve_{update}.npz");curve_rows.append(c)
        history=old["history"];observations=old["observations"];caches=old["caches"];steps=0;restore(a,Path(replay)/"adapter.pt")
    gate=fitting_gate(curve_rows[-1]["records"],initial_curve["records"])
    assert digest(d.model)==frozen and all(p.grad is None for p in d.model.parameters())
    result=dict(arm=arm,capacity=capacity,initial_digest=initial,final_digest=digest(a),frozen_digest=frozen,
        parameters=sum(p.numel() for p in a.parameters()),optimizer_steps=steps,curves=curve_rows,fit=gate,history=history,observations=observations,actual_readouts=actual,replay_checks=checks,caches=caches,
        files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and p.suffix in (".pt",".npz")},
        wall_seconds=time.monotonic()-started,operational_qualified=False)
    if replay:assert result["final_digest"]==old["final_digest"]
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info,teacher_data_sha256=sha(root/"teachers/protocol.json"),target_scope="COUPLED_MODEL_GENERATED_139_NOT_CONTACT_TRUTH"))
    a.close();print("R27_TRAIN_DONE",arm,steps,gate,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["P","P2","Q0","Q1"],required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.arm,a.replay_from)

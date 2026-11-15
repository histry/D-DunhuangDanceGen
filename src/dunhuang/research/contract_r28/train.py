"""Fixed-dose real-source pair. Only the input/target coordinate-C4 package differs."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from .representation import packet
from dunhuang.research.state_r27.common import read,sha,write,fresh,forbid_optimizers,runtime,digest,checkpoint,restore,vector,gradient_norms,readout_check,metrics,commit,same_arrays
from dunhuang.research.state_r27.trace import row_noise
from dunhuang.research.state_r27.qualification import fitting_gate

def dataset(root,cases):
    root=Path(root);protocol=read(root/"data/protocol.json")
    rows=read(root/"data/result.json")["records"];result=[]
    for c in cases:
        row=next(r for r in rows if r["case"]==c);path=root/"data"/row["arrays"]
        assert sha(path)==protocol["files"][row["arrays"]]
        result.append(dict(np.load(path)))
    return result

def loss_terms(pred,y,base):
    parts=dict(body=(pred[:,64:160,7:]-y[:,64:160,7:]).square().mean(),
        root=(pred[:,64:160,4:7]-y[:,64:160,4:7]).square().mean(),
        velocity=(pred[:,65:160,7:]-pred[:,64:159,7:]-y[:,65:160,7:]+y[:,64:159,7:]).square().mean(),
        outside=torch.cat((pred[:,:64]-base[:,:64],pred[:,160:]-base[:,160:]),1).square().mean())
    # Neither arm calls placeholder/height C4 measured contact truth. Same C4-loss mask in both arms.
    return parts["body"]+.1*parts["root"]+.1*parts["velocity"]+.25*parts["outside"],parts

def objective(d,a,y,music,ref,t,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);ts=torch.full((4,),t,device=y.device,dtype=torch.long)
    xt=d.q_sample(y,ts,torch.randn_like(y));a.clear();rng=torch.cuda.get_rng_state();cpu=torch.get_rng_state()
    with torch.no_grad():base=d.model(xt,music,None,ts,cond_drop_prob=.2)
    torch.cuda.set_rng_state(rng);torch.set_rng_state(cpu);a.set_reference(**ref)
    pred=d.model(xt,music,None,ts,cond_drop_prob=.2);a.clear()
    loss,parts=loss_terms(pred,y,base)
    return loss,{k:float(v.detach()) for k,v in parts.items()}

def curves(cfg,d,a,n,s,f,rows,contract,step,out):
    y=torch.tensor(np.stack([x[contract+"_target"] for x in rows]),device="cuda");music=torch.tensor(np.stack([x["music"] for x in rows]),device="cuda")
    truth=np.stack([x["model_raw"] for x in rows]);points=np.stack([x["model_points"] for x in rows])
    references=[x["model_raw"][64:160,7:] for x in rows];arrays={};records=[]
    for t in cfg["times"]:
        ts=torch.full((4,),t,device="cuda",dtype=torch.long)
        with row_noise(cfg["real_eval_seeds"][0],device="cuda"):epsilon=torch.randn_like(y)
        xt=d.q_sample(y,ts,epsilon)
        for mode in ("correct","wrong","none"):
            a.clear()
            if mode!="none":a.set_reference(**packet(references if mode=="correct" else references[1:]+references[:1]))
            with torch.no_grad():
                pred=d.model(xt,music,None,ts,cond_drop_prob=0)
                raw=n.unnormalize(pred.clone());p=f(raw,s)[:,:,:22]
            arrays[f"{t}_{mode}"]=pred.cpu().numpy()
            for i in range(4):
                records.append(dict(teacher=i,case=cfg["train_cases"][i],t=t,mode=mode,
                    clipped_mse=float((pred[i,64:160,7:].clamp(-1,1)-y[i,64:160,7:]).square().mean()),
                    raw_mse=float((pred[i,64:160,7:]-y[i,64:160,7:]).square().mean()),
                    metrics=metrics(raw[i].cpu().numpy(),p[i].cpu().numpy(),truth[i],points[i]),
                    root_rmse_m=float((raw[i,64:160,4:7]-torch.tensor(truth[i,64:160,4:7],device="cuda")).square().mean().sqrt()),
                    physical_target_scope="MODEL0_COMMON_COORDINATES_ORIGINAL_REAL_SOURCE; OFFICIAL_CLIPPED_PREDICTION"))
    a.clear();np.savez_compressed(out/f"curve_{step}.npz",**arrays)
    return dict(update=step,records=records)

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out);assert arm in ("T0","T1")
    if replay:forbid_optimizers()
    contract="legacy" if arm=="T0" else "model";rows=dataset(root,cfg["train_cases"])
    d,n,s,f,e,info,a=runtime(read("src/dunhuang/research/state_r27/config.json"))
    initial=digest(a);frozen=digest(d.model);y=torch.tensor(np.stack([x[contract+"_target"] for x in rows]),device="cuda")
    music=torch.tensor(np.stack([x["music"] for x in rows]),device="cuda");ref=packet([x["model_raw"][64:160,7:] for x in rows])
    curves_list=[curves(cfg,d,a,n,s,f,rows,contract,0,out)];observations=[];history=[];checks=[];actual=[];started=time.monotonic()
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
                checkpoint(a,out/f"weights_{i+1}.pt",cfg,arm,i+1);curves_list.append(curves(cfg,d,a,n,s,f,rows,contract,i+1,out))
            if (i+1)%200==0:print("R28_TRAIN",arm,i+1,float(loss.detach()),flush=True)
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
            i=curve["update"];restore(a,Path(replay)/f"weights_{i}.pt");current=curves(cfg,d,a,n,s,f,rows,contract,i,out)
            assert current==curve;same_arrays(out/f"curve_{i}.npz",Path(replay)/f"curve_{i}.npz");curves_list.append(current)
        history=old["history"];observations=old["observations"];steps=0;restore(a,Path(replay)/"adapter.pt")
    assert digest(d.model)==frozen and all(p.grad is None for p in d.model.parameters())
    result=dict(arm=arm,contract=contract,initial_digest=initial,final_digest=digest(a),frozen_digest=frozen,
        parameters=sum(p.numel() for p in a.parameters()),optimizer_steps=steps,curves=curves_list,fit=fitting_gate(curves_list[-1]["records"],curves_list[0]["records"]),
        history=history,observations=observations,actual_readouts=actual,replay_checks=checks,per_target_presentations=1200,
        files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        Core_C4_loss_applied=False,full_139_input=True,reference_representation="CANONICAL_SO3_ROWS",wall_seconds=time.monotonic()-started,operational_qualified=False)
    if replay:assert result["final_digest"]==old["final_digest"]
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info))
    a.close();print("R28_TRAIN_DONE",arm,steps,result["fit"]["passed"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",required=True,choices=["T0","T1"]);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.root,args.out,args.arm,args.replay_from)

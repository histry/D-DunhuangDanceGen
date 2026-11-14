"""Two matched small state heads, fixed budget, frozen R13 and no model selection."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.masked_denoiser.common import tensor_hash
from .engine import load,tensors,observed
from .contract import metadata
from .model import fk
from .objective import losses,readout

def digest(module):
    h=hashlib.sha256()
    for name,p in sorted(module.state_dict().items()):h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def run(cfg,data_root,out,mode,replay=None):
    out=fresh(out);assert mode in ("cross","self");pre=read(Path(data_root)/"protocol.json")
    population=read(Path(data_root)/"result.json");assert population["continuous_data_ready"]
    fit=[r for r in population["records"] if r["role"]=="fit"];allrows=population["records"]
    old=read(Path(replay)/"result.json") if replay else None
    if replay:forbid_optimizers()
    wrapper,diffusion,data,predict,head=load(cfg,mode,old["checkpoint"] if replay else None)
    base_digest=wrapper.digest();initial_digest=old["initial_digest"] if replay else digest(head)
    parity=np.load(checked(pre["parity"]));parents=list(map(int,parity["parents"]))
    offsets=torch.tensor(parity["offsets"],device="cuda",dtype=torch.float32)
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    cached={};parity_errors=[]
    for r in allrows:
        z,y,m=tensors(r);origin=torch.tensor(z["origin"][None],device="cuda")
        with torch.no_grad():error=float(abs(fk(y,scale,bias,origin,parents,offsets)[0].cpu().numpy()-parity[r["role"]+"_"+str(r["case"])]).max())
        assert error<=1e-5;parity_errors.append(error);cached[r["role"],r["case"]]=(z,y,m,origin)
    history=[];ledger=[];updates=0
    if not replay:
        opt=torch.optim.AdamW(head.parameters(),lr=cfg["learning_rate"]);rng=np.random.default_rng(cfg["seed"])
        # Initial head must preserve the frozen denoiser exactly.
        z,y,m,o=cached["fit",0];v,k,h=observed(y,"free");meta=torch.tensor(metadata()[None],device="cuda")
        t=torch.tensor([519],device="cuda");torch.manual_seed(cfg["seed"]+200);xt=diffusion.q_sample(y,t)
        with torch.no_grad():
            p,_=predict(xt,m,v,k,meta,t);a,_=predict(xt,m,v,k,meta,t,"removed");assert torch.equal(p,a)
            poison=v.clone();poison[~k]=float("nan");b,_=predict(xt,m,poison,k,meta,t);assert torch.equal(p,b)
        for step in range(cfg["training_steps"]):
            case=int(rng.integers(0,len(fit)));z,y,m,origin=cached["fit",case]
            boundary=int(rng.integers(80,145));task=("free","inbetween","sparse","progressive")[step%4]
            extra=[(int(rng.integers(8,215)),int(rng.integers(12,20))) for _ in range(4)] if task=="sparse" else None
            v,k,h=observed(y,"free" if task=="sparse" else task,hard=(step//4)%2==0,
                prefix=y[0,:75].cpu().numpy() if task=="progressive" else None,boundary=boundary,extra=extra)
            meta=torch.tensor(metadata(boundary=boundary)[None],device="cuda")
            torch.manual_seed(cfg["seed"]+10000+step);torch.cuda.manual_seed_all(cfg["seed"]+10000+step)
            t=torch.tensor([int(rng.integers(0,1000))],device="cuda");xt=diffusion.q_sample(y,t)
            pred,_=predict(xt,m,v,k,meta,t);parts=losses(pred,y,v,k,h,scale,bias,origin,parents,offsets)
            context=pred.sum()*0
            if step%4==0:
                wrong,_=predict(xt,m,v,k,meta,t,"reversed")
                context=torch.relu(cfg["context_margin"]+parts["reconstruction"]-(wrong[:,:,4:]-y[:,:,4:]).square().mean())
            parts["context"]=context
            objective=sum(cfg["loss_weights"][name]*value for name,value in parts.items())
            if not torch.isfinite(objective):raise ValueError("Nonfinite registered training objective")
            opt.zero_grad();objective.backward()
            assert all(p.grad is None for p in wrapper.parameters())
            norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True));opt.step();updates+=1
            history.append(dict(step=step,loss=float(objective.detach()),gradient_norm=norm,parts={n:float(v.detach()) for n,v in parts.items()}))
            ledger.append(dict(step=step,case=case,group=fit[case]["left"]["source_group"],task=task,
                boundary=boundary,hard=bool(h.any()),t=int(t[0]),x_t_sha256=tensor_hash(xt),known_sha256=tensor_hash(k)))
            if (step+1)%40==0:print("R17_TRAIN",mode,step+1,history[-1]["loss"],flush=True)
    q={};scores=[]
    with torch.no_grad():
        for r in allrows:
            z,y,m,origin=cached[r["role"],r["case"]]
            for task in ("free","inbetween"):
                v,k,h=observed(y,task);meta=torch.tensor(metadata()[None],device="cuda")
                for timestep in (519,19):
                    seed=cfg["seed"]+50000+(0 if r["role"]=="fit" else 1000)+r["case"]*100+timestep
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);t=torch.tensor([timestep],device="cuda");xt=diffusion.q_sample(y,t)
                    for control in ("correct","reversed","removed"):
                        p,_=predict(xt,m,v,k,meta,t,control)
                        key=r["role"]+"_"+str(r["case"])+"_"+task+"_"+str(timestep)+"_"+control
                        q[key]=p[0].cpu().numpy()
                        scores.append(dict(role=r["role"],case=r["case"],task=task,t=timestep,control=control,
                            **readout(p,y,scale,bias,origin,parents,offsets)))
    np.savez_compressed(out/"q_eval.npz",**q)
    assert wrapper.digest()==base_digest
    if replay:
        assert scores==old["q_scores"] and digest(head)==old["final_digest"]
        same_arrays(out/"q_eval.npz",Path(replay)/"q_eval.npz");checkpoint=old["checkpoint"]
    else:
        path=out/"state.pt";torch.save(dict(schema=cfg["schema"],config=cfg,mode=mode,state=head.state_dict(),
            base_digest=base_digest,code_commit=commit(),operational_qualified=False),path);checkpoint=ref(path)
    result=dict(mode=mode,checkpoint=checkpoint,parameters=sum(p.numel() for p in head.parameters()),optimizer_steps=updates,
        initial_digest=initial_digest,final_digest=digest(head),base_digest=base_digest,base_unchanged=True,q_scores=scores,
        history=history,ledger=ledger,fk_parity_max_m=max(parity_errors),new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,data=ref(Path(data_root)/"protocol.json"),
        fit=[r["left"] for r in fit],eval=[r["left"] for r in allrows if r["role"]=="eval"],
        fixed_budget=True,validation_selection=False,teacher_forcing="Train progressive uses real past75; evaluation must use generated prefix"))
    predict.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--mode",choices=["cross","self"],required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.data,a.out,a.mode,a.replay_from)

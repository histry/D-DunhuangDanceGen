"""Read-only Train objective attribution after fixed-budget failure, not a direction search."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.global_clock_r17.engine import tensors,observed
from dunhuang.research.global_clock_r17.contract import metadata
from dunhuang.research.global_clock_r17.objective import losses
from dunhuang.research.masked_denoiser.common import tensor_hash
from dunhuang.research.global_clock_r17.train import digest
from .engine import load,population,trainable,unknown_mse,amplitude

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pre,fit,ev=population(cfg)
    fit=[r for r in fit if r["case"] in cfg["tiny_cases"]]
    parity=np.load(checked(pre["parity"]));parents=list(map(int,parity["parents"]))
    offsets=torch.tensor(parity["offsets"],device="cuda");records=[];arrays={}
    labels=["reconstruction","boundary","world_fk","amplitude","context","unknown_reference"]
    models={}
    for mode in ("self","unique"):
        reference=read(root/"tiny"/mode/"result.json")["checkpoint"]
        w,d,data,predict,head=load(cfg,mode,reference);before=w.digest();head_before=digest(head)
        params=trainable(w,head);split=sum(p.numel() for p in w.selected())
        scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
        for row in fit:
            z,y,m=tensors(row);origin=torch.tensor(z["origin"][None],device="cuda")
            for task_index,task in enumerate(cfg["tasks"]):
                v,k,h=observed(y,task,prefix=y[0,:75].cpu().numpy() if task=="progressive" else None)
                meta=torch.tensor(metadata()[None],device="cuda")
                for timestep in (519,19):
                    seed=cfg["seed"]+40000+row["case"]*100+task_index*1000+timestep
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    t=torch.tensor([timestep],device="cuda");xt=d.q_sample(y,t)
                    pred,_=predict(xt,m,v,k,meta,t);wrong,_=predict(xt,m,v,k,meta,t,"content")
                    terms=losses(pred,y,v,k,h,scale,bias,origin,parents,offsets)
                    terms["amplitude"]=amplitude(pred,y,scale,bias,origin,parents,offsets)
                    terms["context"]=torch.relu(cfg["context_margin"]+unknown_mse(pred,y,k)-unknown_mse(wrong,y,k))
                    terms["unknown_reference"]=unknown_mse(pred,y,k)
                    vectors=[]
                    for label in labels:
                        weighted=terms[label]*(cfg["loss_weights"].get(label,1.))
                        grads=torch.autograd.grad(weighted,params,retain_graph=True,allow_unused=True)
                        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,grads)]))
                    V=torch.stack(vectors);assert torch.isfinite(V).all()
                    key=f"{mode}_{row['case']}_{task}_{timestep}"
                    grams={}
                    for group,a,b in (("motion",0,split),("head",split,V.shape[1])):
                        G=V[:,a:b]@V[:,a:b].T
                        norms=G.diag().clamp_min(0).sqrt();cos=G/(norms[:,None]*norms[None,:]).clamp_min(1e-30)
                        arrays[key+"_"+group+"_gram"]=G.cpu().numpy()
                        grams[group]=dict(weighted_gradient_norm=norms.cpu().tolist(),
                            cosine_with_unknown=cos[:,-1].cpu().tolist(),
                            cosine_with_reconstruction=cos[:,0].cpu().tolist())
                    records.append(dict(mode=mode,case=row["case"],task=task,t=timestep,x_t_sha256=tensor_hash(xt),
                        unweighted_losses={k:float(v.detach()) for k,v in terms.items()},groups=grams))
                    del V,vectors,pred,wrong,terms
        assert w.digest()==before and digest(head)==head_before;predict.close();models[mode]=reference
    np.savez_compressed(out/"gradient_grams.npz",**arrays)
    result=dict(labels=labels,records=records,models=models,optimizer_steps=0,
        scope="Local gradients at saved final heads on Fit0/1 only; not optimizer trajectories or causal attribution",
        eval_used=False,new_training=False,operational_qualified=False)
    if replay:
        same_arrays(out/"gradient_grams.npz",Path(replay)/"gradient_grams.npz")
        assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        parent_train={m:ref(root/"tiny"/m/"result.json") for m in ("self","unique")},
        fixed_states="Fit0/1 x three tasks x t519/19",no_candidates=True,no_parameter_updates=True))
    print("R18_OBJECTIVE_DIAG",len(records),"optimizer_steps=0",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)

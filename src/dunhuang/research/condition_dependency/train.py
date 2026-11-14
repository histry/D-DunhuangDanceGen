"""Fixed C0/C1 contrast: same geometry/parameters/batches/CFG, only Root-response loss differs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data,load_model,digest,masks,fixed_rows
from dunhuang.research.boundary_context.train import loss as boundary_loss
from .adapter import ContentContextAdapter

def shifted_context(target,known,a,b,scale,axis,delta):
    v=target.clone();v[:,b:b+8,4+axis]+=scale[4+axis]*delta
    valid=bool((abs(v[known])<=1+32*torch.finfo(v.dtype).eps).all())
    u=torch.arange(1,b-a+1,device=v.device,dtype=v.dtype)/(b-a+1);shape=u**3*(10+u*(-15+6*u))
    expected=torch.zeros((len(v),b-a,3),device=v.device);expected[:,:,axis]=shape[None]*scale[4+axis]*delta
    return v,expected,valid

def run(cfg,audit,out,arm,replay=None):
    out=fresh(out);audit=Path(audit)
    assert read(audit/"p0/result.json")["new_optimizer_steps"]==0
    pf=Path(cfg["preflight"]);pp=read(pf/"protocol.json");assets=read(checked(pp["assets"]));rows=read(pf/"rows.json")
    assert read(pf/"result.json")["bridge_training_data_ready"]
    data=Data(np.load(checked(assets["scaler"])));fit=[r for r in rows if r["role"]=="train"];val=fixed_rows(rows,"val",cfg["validation_windows"],cfg["seed"])
    assert set(r["source_group"] for r in fit).isdisjoint(r["source_group"] for r in val)
    model,diffusion,norm=load_model(assets);before=digest(model);scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    if arm=="C2":
        from .balanced import BalancedContextAdapter
        stats=read(Path(cfg["units"])/"result.json")
        assert stats["role"]=="train" and stats["training_windows"]==1833 and stats["validation_tensor_reads"]==0
        inv=np.load(Path(cfg["units"])/"inv_rms.npy");assert np.array_equal(inv,np.asarray(stats["inv_rms"],np.float32))
        adapter=BalancedContextAdapter(data.scale,data.bias,inv).cuda()
    else:adapter=ContentContextAdapter(data.scale,data.bias).cuda()
    state={}
    def hook(module,args,music):
        return music+adapter(state["value"],state["known"],music,state["t"]) if state["enabled"] else music
    handle=model.cond_encoder.register_forward_hook(hook)
    def unconditional(xt,music,t):
        state["enabled"]=False
        with torch.no_grad():return model(xt,music,t,cond_drop_prob=1.)
    def predict(xt,music,t,value,known,unc,control="observed"):
        if control=="removed":known=torch.zeros_like(known)
        if control=="wrong":value=value.roll(1,0)
        state.update(enabled=control!="zero",value=value,known=known,t=t)
        conditional=model(xt,music,t,cond_drop_prob=0.)
        return (unc+2*(conditional-unc)).clamp(-1,1)
    protocol=dict(code_commit=commit(),config=cfg,arm=arm,preflight=ref(pf/"protocol.json"),rows=ref(pf/"rows.json"),audit=ref(audit/"p0/protocol.json"),
        objective="real masked Root/rotation + boundary loss; C1/C2 add explicitly synthetic horizontal Root state response only",
        synthetic_response_is_not_artistic_or_contact_truth=True,validation_selection=False,full_backbone_frozen=True,cfg_weight=2)
    if arm=="C2":
        parent=Path(cfg["frozen_parent"]);pc=read(parent/"protocol.json")["config"]
        for key in ("seed","data_seed","steps","learning_rate","validation_windows","timesteps","root_shift_m","response_weight","preflight"):
            assert cfg[key]==pc[key]
        protocol["frozen_parent"]=ref(parent/"result.json")
        protocol["feature_units"]=dict(result=ref(Path(cfg["units"])/"result.json"),array=ref(Path(cfg["units"])/"inv_rms.npy"))
    write(out/"protocol.json",protocol)
    # Engineering admission is not a claim of learned quality.
    test,music=data.batch(val[:2]);known,unknown=masks(test,60,90);t=torch.tensor([900,999],device="cuda")
    assert torch.equal(adapter(test,torch.zeros_like(known),torch.zeros(2,150,512,device="cuda"),t),torch.zeros(2,150,512,device="cuda"))
    gate=adapter(test,known,torch.zeros(2,150,512,device="cuda"),t)
    # At a zero head, content features must supply a finite nonzero head-learning Jacobian.
    gate.sum().backward()
    jac=float(adapter.output.weight.grad.norm());assert jac>0 and torch.isfinite(adapter.output.weight.grad).all();adapter.zero_grad()
    write(out/"interface_gate.json",dict(empty_context_exact_zero=True,content_parameter_jacobian_norm=jac,semantic_quality_proven=False))
    history=[];skips=0;updates=0
    if replay:
        payload=torch.load(checked(read(Path(replay)/"result.json")["checkpoint"]),map_location="cpu",weights_only=True);adapter.load_state_dict(payload["state"],strict=True)
    else:
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg["learning_rate"]);rng=np.random.default_rng(cfg["seed"])
        for step in range(cfg["steps"]):
            chosen=[fit[i] for i in rng.integers(0,len(fit),2)];target,music=data.batch(chosen);a=int(rng.integers(60,67));b=int(rng.integers(84,91))
            observed=None if rng.random()<.25 else rng.random(24)<.8;known,unknown=masks(target,a,b,observed)
            if rng.random()<.5:t=torch.tensor(rng.choice([900,999],2),device="cuda")
            else:t=torch.tensor(rng.integers(0,1000,2),device="cuda")
            xt=diffusion.q_sample(target,t);unc=unconditional(xt,music,t);adapter.train()
            pred=predict(xt,music,t,target,known,unc);objective=boundary_loss(pred,target,unknown,a,b,(scale,bias))
            pert,expected,valid=shifted_context(target,known,a,b,scale,step%2,cfg["root_shift_m"]*(1 if step%4<2 else -1))
            cf=torch.zeros((),device="cuda")
            if not valid:skips+=1
            elif arm in ("C1","C2"):
                ppred=predict(xt,music,t,pert,known,unc)
                denom=float(abs(expected).max().detach())+1e-6
                cf=(((ppred[:,a:b,4:7]-pred[:,a:b,4:7]-expected)/denom)**2).mean()
                objective=objective+cfg["response_weight"]*cf
            if not torch.isfinite(objective):raise ValueError("Nonfinite R10 objective")
            optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            assert all(p.grad is None for p in model.parameters())
            history.append(dict(loss=float(objective.detach()),synthetic_response_loss=float(cf.detach()),response_supported=valid))
            if (step+1)%25==0:print("R10_TRAIN",arm,step+1,history[-1],flush=True)
        updates=cfg["steps"]
    scores=[];adapter.eval()
    with torch.no_grad():
        for control in ("zero","observed","removed","wrong","shifted"):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(cfg["seed"]+71);torch.cuda.manual_seed_all(cfg["seed"]+71)
                for k in range(0,len(val),2):
                    target,music=data.batch(val[k:k+2]);known,unknown=masks(target,60,90)
                    for timestep in cfg["timesteps"]:
                        t=torch.full((len(target),),timestep,device="cuda",dtype=torch.long);xt=diffusion.q_sample(target,t);unc=unconditional(xt,music,t)
                        base=predict(xt,music,t,target,known,unc,control if control!="shifted" else "observed")
                        row=dict(control=control,batch=k,t=timestep,masked_cfg_mse=float((base-target)[unknown].square().mean()))
                        if control=="shifted":
                            value,expected,valid=shifted_context(target,known,60,90,scale,0,cfg["root_shift_m"])
                            shifted=predict(xt,music,t,value,known,unc)
                            row.update(shift_supported=valid,response_root_error_m=float((((shifted[:,60:90,4:7]-base[:,60:90,4:7]-expected)/scale[4:7])**2).mean().sqrt()),
                                response_root_rms_m=float((((shifted[:,60:90,4:7]-base[:,60:90,4:7])/scale[4:7])**2).mean().sqrt()))
                        scores.append(row)
    assert digest(model)==before
    if replay:
        old=read(Path(replay)/"result.json");assert old["validation"]==scores;checkpoint=old["checkpoint"]
    else:
        torch.save(dict(schema="r10_balanced_context_v1" if arm=="C2" else "r10_content_context_v1",state=adapter.state_dict(),config=cfg,arm=arm,base_sha256=assets["checkpoint"]["sha256"],code_commit=commit(),operational_qualified=False),out/"adapter.pt");checkpoint=ref(out/"adapter.pt")
    result=dict(arm=arm,history=history,optimizer_steps=updates,checkpoint=checkpoint,validation=scores,cf_support_skips=skips,
        parameters=sum(p.numel() for p in adapter.parameters()),backbone_digest=before,backbone_updated=False,operational_qualified=False,new_whole_song_generations=0)
    write(out/"result.json",result);handle.remove();print("R10_TRAIN_DONE",arm,updates,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--audit",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["C0","C1","C2"],required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.audit,a.out,a.arm,a.replay_from)

"""Finite temporal nonlinear reference decoding; oracle Root is metric context, not generation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import runtime,rows,packet
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,digest,vector,readout_check,forbid_optimizers,metrics,same_arrays

class Decoder(nn.Module):
    def __init__(self,channels,width):
        super().__init__();self.input=nn.Linear(channels,width)
        self.temporal=nn.ModuleList([nn.Conv1d(width,width,3,padding=1) for _ in range(2)])
        self.output=nn.Linear(width,132)
    def forward(self,x):
        h=torch.nn.functional.silu(self.input(x))
        for conv in self.temporal:h=h+torch.nn.functional.silu(conv(h.transpose(1,2)).transpose(1,2))
        return self.output(h)

def save(model,path):torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},path)
def restore(model,path):model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
def objective(model,x,y):
    p=model(x);return (p-y).square().mean()+.1*(p[:,1:]-p[:,:-1]-y[:,1:]+y[:,:-1]).square().mean()

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    d,n,s,f,e,info,a=runtime(cfg,0)
    train,_=rows(cfg,root,"aist","train");held,_=rows(cfg,root,"aist","holdout");data=train+held
    ref=packet(data);mems={};base=digest(d.model)
    mems["raw"]=torch.cat((ref["features"].flatten(2),ref["valid"].float(),ref["clock"]),-1).detach()
    for name in ("C1","T1"):
        if name=="T1":
            from dunhuang.research.state_r27.common import restore as restore_adapter
            restore_adapter(a,Path(cfg["r28"])/"models/T1/adapter.pt")
        with torch.no_grad():a.set_reference(**ref);m=a.context[0]
        mems[name]=torch.cat((m[:,1:],m[:,:1].expand(-1,96,-1)),-1).detach();a.clear()
    y=torch.tensor(np.stack([x["model_raw"][64:160,7:] for x in data]),device="cuda")
    records={};arrays={};allchecks=[]
    for name,x in mems.items():
        folder=out/name;folder.mkdir();torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
        model=Decoder(x.shape[-1],cfg["readout_width"]).cuda();initial=digest(model)
        observations=[];actual=[];curves=[]
        def measure(step):
            with torch.no_grad():p=model(x)
            results=[]
            for i,row in enumerate(data):
                raw=row["model_raw"].copy();raw[64:160,7:]=p[i].cpu().numpy()
                with torch.no_grad():points=f(torch.tensor(raw[None],device="cuda"),s)[0,:,:22].cpu().numpy()
                er=metrics(raw,points,row["model_raw"],row["model_points"])
                v=(np.diff(points[64:160]-points[64:160,:1],axis=0)-np.diff(row["model_points"][64:160]-row["model_points"][64:160,:1],axis=0))*30
                results.append(dict(case=(cfg["train_cases"]+cfg["holdout_cases"])[i],role="train" if i<4 else "readout_holdout_R26_exposed",metrics=er,
                    root_relative_FK_velocity_rmse_mps=float(np.sqrt(np.mean(v**2))),Root_and_outside_Core="SOURCE_ORACLE_FOR_DIAGNOSTIC_ONLY"))
            arrays[f"{name}_{step}"]=p.cpu().numpy();return dict(update=step,records=results)
        curves.append(measure(0))
        if not replay:
            save(model,folder/"initial.pt");opt=torch.optim.AdamW(model.parameters(),lr=cfg["readout_lr"])
            for step in range(cfg["readout_steps"]):
                watch=step in cfg["readout_snapshots"]
                if watch:save(model,folder/f"{step}_pre.pt");before=vector(model).clone()
                opt.zero_grad(set_to_none=True);loss=objective(model,x[:4],y[:4]);loss.backward()
                gradient=float(sum(p.grad.square().sum() for p in model.parameters()).sqrt())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                if watch:
                    save(model,folder/f"{step}_post.pt");observations.append(dict(step=step,loss=float(loss.detach()),gradients={"decoder":gradient},update_norm=float((vector(model)-before).norm())))
                if step+1 in (300,600):save(model,folder/f"weights_{step+1}.pt");curves.append(measure(step+1))
            save(model,folder/"decoder.pt");steps=cfg["readout_steps"]
        else:
            old=read(Path(replay)/"result.json")["models"][name];assert initial==old["initial_digest"]
            for fn,h in old["files"].items():assert sha(Path(replay)/name/fn)==h
            for item in old["observations"]:
                step=item["step"];restore(model,Path(replay)/name/f"{step}_pre.pt");before=vector(model).clone();model.zero_grad(set_to_none=True)
                loss=objective(model,x[:4],y[:4]);loss.backward();gradient=float(sum(p.grad.square().sum() for p in model.parameters()).sqrt())
                restore(model,Path(replay)/name/f"{step}_post.pt")
                value=dict(step=step,loss=float(loss.detach()),gradients={"decoder":gradient},update_norm=float((vector(model)-before).norm()))
                allchecks.append(readout_check(value,item));actual.append(value)
            for oldcurve in old["curves"][1:]:
                step=oldcurve["update"];restore(model,Path(replay)/name/f"weights_{step}.pt");c=measure(step);assert c==oldcurve;curves.append(c)
            restore(model,Path(replay)/name/"decoder.pt");steps=0;observations=old["observations"]
        records[name]=dict(initial_digest=initial,final_digest=digest(model),parameters=sum(p.numel() for p in model.parameters()),input_channels=x.shape[-1],
            optimizer_steps=steps,curves=curves,observations=observations,actual_readouts=actual,
            files=old["files"] if replay else {p.name:sha(p) for p in folder.glob("*.pt")})
        print("R29_TEMPORAL_READOUT",name,steps,flush=True)
    assert digest(d.model)==base and all(p.grad is None for p in a.parameters())
    path=out/"arrays.npz";np.savez_compressed(path,**arrays)
    result=dict(models=records,decoder_optimizer_steps=sum(x["optimizer_steps"] for x in records.values()),generator_optimizer_steps=0,
        replay_checks=allchecks,frozen_encoder_gradients_absent=True,readout_scope="NONLINEAR_DECODABILITY_NOT_INFORMATION_IMPOSSIBILITY_OR_GENERATOR_QUALIFICATION",
        Root_supplied_only_to_FK_metric=True,operational_qualified=False)
    if replay:
        same_arrays(path,Path(replay)/path.name)
        assert all(records[k]["final_digest"]==read(Path(replay)/"result.json")["models"][k]["final_digest"] for k in records)
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),arrays_sha256=sha(path),checkpoint=info));a.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");args=p.parse_args();run(read(args.config),args.root,args.out,args.replay_from)

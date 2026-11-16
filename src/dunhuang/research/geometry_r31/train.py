"""Matched LegacyIO/CanonicalIO adaptation; no new physical losses or data selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import sources
from .interface import runtime
from dunhuang.research.time_r29.common import packet
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,digest,checkpoint,restore,vector,readout_check,forbid_optimizers,same_arrays
from dunhuang.research.contract_r28.train import objective
from dunhuang.research.state_r27.trace import row_noise
from dunhuang.research.continuity_r30.metrics import measure

def grad_norms(bundle):
    groups={"encoder":("reference.parts","reference.input","reference.temporal","reference.norm"),
            "reference_output":("reference.out",),"qkv":("reference.q","reference.k","reference.v"),
            "action_input":("io.input_delta",),"action_output":("io.output_delta",)}
    return {k:float(sum((p.grad.square().sum() for name,p in bundle.named_parameters() if name.startswith(prefix) and p.grad is not None),torch.tensor(0.,device="cuda")).sqrt()) for k,prefix in groups.items()}

def curves(cfg,d,b,n,s,f,rows,target,step,out):
    music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda");records=[];arrays={}
    seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0]
    for t in cfg["times"]:
        ts=torch.full((4,),t,device="cuda",dtype=torch.long)
        with row_noise(seeds,device="cuda"):epsilon=torch.randn_like(target)
        xt=d.q_sample(target,ts,epsilon)
        for mode in ("correct","swapped","none"):
            b.clear()
            if mode!="none":b.set_reference(**packet(rows,mode))
            with torch.no_grad():
                p=d.model(xt,music,None,ts,cond_drop_prob=0);raw=n.unnormalize(p.clone());points=f(raw,s)[:,:,:22]
            arrays[f"{t}_{mode}"]=p.cpu().numpy()
            for i,row in enumerate(rows):
                records.append(dict(case=cfg["train_cases"][i],t=t,mode=mode,
                    clipped_mse=float((p[i,64:160,7:].clamp(-1,1)-target[i,64:160,7:]).square().mean()),
                    encoding_MSE_is_within_contract_only=True,
                    **measure(raw[i].cpu().numpy(),points[i].cpu().numpy(),row["model_raw"],row["model_points"])))
    b.clear();np.savez_compressed(out/f"curve_{step}.npz",**arrays);return dict(update=step,records=records)

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    assert read(root/"qualification/result.json")["new_contract_training_admitted"]
    rows,ledger,old=sources(cfg);data={r["case"]:r["data"] for r in rows if r["domain"]=="aist" and r["role"]=="train"}
    canonical=arm=="CanonicalIO";d,n,s,f,e,info,b=runtime(cfg,canonical)
    initial=digest(b);reference_initial=digest(b.reference);frozen=digest(d.model)
    assert reference_initial==old["initial_digest"] and frozen==old["frozen_digest"]
    with torch.no_grad():targets={c:n.normalize(torch.tensor(row["model_raw"][None],device="cuda"))[0] for c,row in data.items()}
    probes=[data[c] for c in cfg["train_cases"]];probe_target=torch.stack([targets[c] for c in cfg["train_cases"]])
    cs=[curves(cfg,d,b,n,s,f,probes,probe_target,0,out)];history=[];obs=[];actual=[];checks=[]
    def loss(row):
        v=[data[c] for c in row["cases"]];y=torch.stack([targets[c] for c in row["cases"]]);m=torch.tensor(np.stack([r["music"] for r in v]),device="cuda")
        return objective(d,b,y,m,packet(v),row["t"],row["seed"])
    if not replay:
        (out/"snapshots").mkdir();checkpoint(b,out/"initial.pt",cfg,arm,0);opt=torch.optim.AdamW(b.parameters(),lr=cfg["learning_rate"])
        for row in ledger:
            i=row["step"];watch=i in cfg["snapshot_steps"]
            if watch:before=vector(b).clone();checkpoint(b,out/f"snapshots/{i}_pre.pt",cfg,arm,i)
            opt.zero_grad(set_to_none=True);value,parts=loss(row);assert torch.isfinite(value);value.backward()
            gradients=grad_norms(b) if watch else None
            torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(row,loss=float(value.detach()),parts=parts))
            if watch:
                checkpoint(b,out/f"snapshots/{i}_post.pt",cfg,arm,i+1);obs.append(dict(step=i,loss=float(value.detach()),parts=parts,gradients=gradients,update_norm=float((vector(b)-before).norm())))
            if i+1 in cfg["curve_updates"]:
                checkpoint(b,out/f"weights_{i+1}.pt",cfg,arm,i+1);cs.append(curves(cfg,d,b,n,s,f,probes,probe_target,i+1,out))
            if (i+1)%200==0:print("R31_TRAIN",arm,i+1,float(value.detach()),flush=True)
        checkpoint(b,out/"adapter.pt",cfg,arm,1200);steps=1200
    else:
        old_result=read(Path(replay)/"result.json");assert initial==old_result["initial_digest"]
        for name,h in old_result["files"].items():assert sha(Path(replay)/name)==h
        for item in old_result["observations"]:
            i=item["step"];restore(b,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(b).clone();b.zero_grad(set_to_none=True)
            value,parts=loss(ledger[i]);value.backward();g=grad_norms(b);restore(b,Path(replay)/f"snapshots/{i}_post.pt")
            current=dict(step=i,loss=float(value.detach()),parts=parts,gradients=g,update_norm=float((vector(b)-before).norm()))
            checks.append(readout_check(current,item));actual.append(current);write(out/f"actual_{i}.json",current)
        for oldcurve in old_result["curves"][1:]:
            step=oldcurve["update"];restore(b,Path(replay)/f"weights_{step}.pt");c=curves(cfg,d,b,n,s,f,probes,probe_target,step,out)
            assert c==oldcurve;cs.append(c);same_arrays(out/f"curve_{step}.npz",Path(replay)/f"curve_{step}.npz")
        restore(b,Path(replay)/"adapter.pt");history=old_result["history"];obs=old_result["observations"];steps=0
    assert digest(d.model)==frozen and all(p.grad is None for p in d.model.parameters())
    result=dict(arm=arm,canonical=canonical,parameters=sum(p.numel() for p in b.parameters()),reference_initial_digest=reference_initial,
        initial_digest=initial,final_digest=digest(b),frozen_digest=frozen,optimizer_steps=steps,history=history,curves=cs,
        observations=obs,actual_readouts=actual,replay_checks=checks,per_source_presentations=300,recordings=16,music_groups=8,
        files=old_result["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        physical_losses_added=False,phase_inputs_zero=True,new_model_contract=True,operational_qualified=False,full_native_status="NOT_EVALUABLE")
    if replay:assert result["final_digest"]==old_result["final_digest"]
    b.close();write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,qualification_sha256=sha(root/"qualification/result.json")))
    print("R31_TRAIN_DONE",arm,steps,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["LegacyIO","CanonicalIO"],required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.arm,a.replay_from)

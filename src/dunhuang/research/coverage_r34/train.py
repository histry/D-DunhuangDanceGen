"""Matched legacy versus continuous coverage; the prior and general route are fixed."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from .common import *
from .common import Runtime,save as save_change,restore as restore_change
from .data import packet
from dunhuang.research.control_r33.objective import partial_forward
from .objective import forward as coverage_forward

from dunhuang.research.contract_r28.train import loss_terms
from dunhuang.research.content_r32.common import local_metrics

def gradients(module):
    groups={"encoder":("parts","input","temporal","norm"),"injection":("q.","k.","v.","out.","qphase.")}
    return {g:float(sum((p.grad.square().sum() for n,p in module.named_parameters() if n.startswith(prefix) and p.grad is not None),torch.tensor(0.,device="cuda")).sqrt()) for g,prefix in groups.items()}

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    calibration=read(Path(cfg["r33"])/"calibration/result.json")
    admission=read(root/"data/result.json")
    if not admission["training_admitted"]:
        write(out/"result.json",dict(arm=arm,admitted=False,optimizer_steps=0,sampler_calls=0,generated_trajectories=0));return
    rt=Runtime(cfg);initial=digest(rt.c.change);pop,_,data,z=population_data()
    source_arrays=dict(np.load(root/"data/arrays.npz"))
    assert sha(root/"data/arrays.npz")==read(root/"data/protocol.json")["files"]["arrays.npz"]
    coverage=arm=="Coverage";trace=np.load(root/("data/Coverage_states.npz" if coverage else "data/Legacy_states.npz"))
    states=dict(xt=trace["xt"][:,[1,2,4,5]],times=trace["times"]);initial_states={k:v.copy() for k,v in states.items()}
    key="body" if coverage else "legacy_body";pk="local_points" if coverage else "legacy_points"
    body=torch.tensor(source_arrays[key][[0,1,0,1]],device="cuda");target=torch.tensor(source_arrays[pk][[0,1,0,1]],device="cuda")
    target=target-target[:,:,:1] # Stored legacy FK has the model pelvis offset; never use it as world truth.
    music=torch.tensor(z["music"][[0,0,1,1]],device="cuda");source_packet=packet(source_arrays,[(0,0),(0,1),(1,0),(1,1)],coverage)
    core_target=body[:,64:160] if coverage else body;core_points_target=target[:,64:160] if coverage else target
    def objective(xt,time,seed):
        if coverage:return coverage_forward(rt,xt,music,source_packet,time,seed,body,target,scales)
        return partial_forward(rt,xt,music,source_packet,time,seed,body,target,cfg32,scales)
    cfg32=read("src/dunhuang/research/content_r32/config.json");scales=calibration["scales"];weight=calibration["weight"]
    parent=Path(cfg["r33"])/"models/Dyn/result.json";assert sha(parent)==read(Path(cfg["r33"])/"artifacts_manifest.json")["files"]["models/Dyn/result.json"]["sha256"]
    history32=read(parent)["history"];schedule=[{k:v for k,v in row.items() if k not in ("loss","parts")} for row in history32]
    counts=Counter(c for row in schedule if row["task"]=="paired" for c in row["cases"]);assert len(counts)==16 and set(counts.values())=={75}
    assert len(schedule)==cfg["train_steps"]==600 and {int(t) for t in states["times"]}=={r["t"] for r in schedule if r["task"]=="partial"}
    arrays={};curves=[];history=[];observations=[];actual=[];checks=[];calls=0;refresh_noise=[]
    def loss(row):
        if row["task"]=="partial":
            xt=torch.tensor(states["xt"][states["times"].tolist().index(row["t"])],device="cuda")
            base,parts,dyn,dp,pred,fk,templates=objective(xt,row["t"],row["seed"])
            return base+weight*dyn,dict({k:float(v.detach()) for k,v in parts.items()},dynamic=float(dyn.detach()),dynamic_D1=float(dp["D1"].detach()),dynamic_D2=float(dp["D2"].detach()))
        rows=[data[c] for c in row["cases"]];y=rt.n.normalize(torch.tensor(np.stack([r["model_raw"] for r in rows]),device="cuda"))
        m=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda");t=torch.full((4,),row["t"],device="cuda",dtype=torch.long)
        torch.manual_seed(row["seed"]);torch.cuda.manual_seed_all(row["seed"]);xt=rt.d.q_sample(y,t,torch.randn_like(y));rt.c.clear_all()
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        with torch.no_grad():teacher=rt.d.model(xt,m,None,t,cond_drop_prob=.2)
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        rt.c.set_source(source([core_body(r) for r in rows],"aist",uids=[f"aist_{c}" for c in row["cases"]],usage="change_replay"))
        pred=rt.d.model(xt,m,None,t,cond_drop_prob=.2);rt.c.clear_all();base,parts=loss_terms(pred,y,teacher)
        return base,{k:float(v.detach()) for k,v in parts.items()}
    def curve(update):
        records=[]
        for t in cfg["times"]:
            xt=torch.tensor(initial_states["xt"][initial_states["times"].tolist().index(t)],device="cuda")
            with torch.no_grad():base,parts,dyn,dp,pred,fk,templates=objective(xt,t,cfg["seed"]+1000+t)
            arrays[f"{update}_{t}"]=pred.cpu().numpy()
            records.append(dict(t=t,base=float(base),dynamic=float(dyn),body=float(parts["body"]),geometry=float(parts["local_geometry"]),
                metrics=[local_metrics(pred[i,64:160,7:],core_target[i].cpu().numpy(),core_points_target[i].cpu().numpy(),rt.s,rt.f) for i in range(4)]))
        return dict(update=update,records=records)
    def refresh(path):
        conditions=[(m,ev) for m in range(2) for ev in (-1,0,1)]
        ref=packet(source_arrays,conditions,coverage)
        pred,nr,tr=sample(rt,torch.tensor(np.stack([z["music"][m] for m,ev in conditions]),device="cuda"),source_packet=ref,common_seed=24041,trace=True)
        prior=np.load(Path(cfg["r33"])/"evaluation/arrays.npz")["train_Dyn_correct_normalized"]
        assert np.array_equal(pred[[0,3]].cpu().numpy(),prior[[0,3]])
        values=trace_arrays(tr);np.savez_compressed(path,**values)
        return dict(xt=values["xt"][:,[1,2,4,5]],times=values["times"]),nr
    curves.append(curve(0))
    if not replay:
        (out/"snapshots").mkdir();save_change(rt,out/"initial.pt",cfg,arm,0);opt=torch.optim.AdamW(rt.c.change.parameters(),lr=cfg["learning_rate"])
        for row in schedule:
            i=row["step"]
            if i==300:save_change(rt,out/"weights_300.pt",cfg,arm,i);states,nr=refresh(out/"refresh_300.npz");calls+=1;refresh_noise.append(nr)
            watch=i in cfg["snapshot_steps"]
            if watch:save_change(rt,out/f"snapshots/{i}_pre.pt",cfg,arm,i);before=vector(rt.c.change).clone()
            opt.zero_grad(set_to_none=True);value,parts=loss(row);assert torch.isfinite(value)
            torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            g=gradients(rt.c.change) if watch else None
            torch.nn.utils.clip_grad_norm_(rt.c.change.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(row,loss=float(value.detach()),parts=parts))
            if watch:
                save_change(rt,out/f"snapshots/{i}_post.pt",cfg,arm,i+1);observations.append(dict(step=i,loss=float(value.detach()),parts=parts,gradients=g,update_norm=float((vector(rt.c.change)-before).norm())))
            if (i+1)%100==0:print("R34_TRAIN",arm,i+1,float(value.detach()),flush=True)
        save_change(rt,out/"adapter.pt",cfg,arm,600);curves.append(curve(600));steps=600
    else:
        old=read(Path(replay)/"result.json");assert initial==old["initial_digest"]
        for fn,h in old["files"].items():assert sha(Path(replay)/fn)==h
        for item in old["observations"]:
            i=item["step"]
            if i>=300:
                values=np.load(Path(replay)/"refresh_300.npz");states=dict(xt=values["xt"][:,[1,2,4,5]],times=values["times"])
            restore_change(rt,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(rt.c.change).clone();rt.c.change.zero_grad(set_to_none=True)
            value,parts=loss(schedule[i]);torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            g=gradients(rt.c.change);restore_change(rt,Path(replay)/f"snapshots/{i}_post.pt")
            v=dict(step=i,loss=float(value.detach()),parts=parts,gradients=g,update_norm=float((vector(rt.c.change)-before).norm()))
            checks.append(readout_check(v,item));actual.append(v)
        restore_change(rt,Path(replay)/"weights_300.pt");states,nr=refresh(out/"refresh_300.npz");calls+=1;same_arrays(out/"refresh_300.npz",Path(replay)/"refresh_300.npz")
        refresh_noise=old["refresh_noise"];restore_change(rt,Path(replay)/"adapter.pt");curves.append(curve(600));assert curves==old["curves"]
        history=old["history"];observations=old["observations"];steps=0
    final=digest(rt.c.change);rt.check();frozen=rt.frozen;rt.close()
    np.savez_compressed(out/"curves.npz",**arrays)
    if replay:assert final==old["final_digest"];same_arrays(out/"curves.npz",Path(replay)/"curves.npz")
    result=dict(arm=arm,admitted=True,initial_digest=initial,final_digest=final,frozen=frozen,dynamic_weight=weight,scales=scales,optimizer_steps=steps,curves=curves,history=history,
        observations=observations,actual_readouts=actual,replay_checks=checks,refresh_noise=refresh_noise,paired_source_counts=dict(counts),trainable_parameters=265328,source_target_frames=256 if coverage else 96,
        general_reference_updated=False,io_updated=False,context_trained=False,partial_Root_C4_truth=False,sampler_calls=calls,generated_trajectories=6*calls,
        auxiliary_state_policy="R34_DYN_ACTUAL_MATCHED_REFERENCE_INITIAL; CURRENT_ARM_REFRESH300; NOT_PER_STEP_ON_POLICY",
        files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        operational_qualified=False,full_native_status="NOT_EVALUABLE",whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,calibration_sha256=sha(Path(cfg["r33"])/"calibration/result.json"),source_data_sha256=sha(root/"data/arrays.npz"),target_point_contract="SUBTRACT_PELVIS_FROM_LEGACY_AND_FULL_FK; WORLD_ROOT_UNKNOWN")))
    print("R34_TRAIN_DONE",arm,steps,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["Extra","Coverage"],required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.arm,a.replay_from)

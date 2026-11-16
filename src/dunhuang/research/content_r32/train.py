"""Frozen-I/O partial-observation pilot, followed only on admission by matched domain trials."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from .partial import partial_objective,partial_loss
from dunhuang.research.contract_r28.train import objective as paired_objective
from dunhuang.research.time_r29.common import packet
from dunhuang.research.state_r27.trace import sample,trace_arrays
from dunhuang.research.state_r27.common import write,fresh,commit,checkpoint,vector,readout_check,forbid_optimizers,same_arrays

def train_gradient(b):
    names=dict(encoder=("parts","input","temporal","norm"),injection=("q.","k.","v.","out.","qphase."))
    return {key:float(sum((p.grad.square().sum() for name,p in b.reference.named_parameters() if name.startswith(prefix) and p.grad is not None),torch.tensor(0.,device="cuda")).sqrt()) for key,prefix in names.items()}

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    readout=read(root/"readout/result.json")
    admitted=readout["partial_training_readout_admitted"]
    if arm!="pilot":admitted=admitted and read(root/"models/pilot/result.json")["main_training_admitted"]
    if not admitted:
        write(out/"result.json",dict(arm=arm,admitted=False,optimizer_steps=0,sampler_calls=0,generated_trajectories=0,operational_qualified=False));return
    pop,ledger,data,z=population(cfg);d,n,s,f,e,info,b=runtime(cfg);frozen=fingerprint(d,b);initial=digest(b)
    aist=[data[c] for c in cfg["train_cases"]];bodyA=np.stack([core_body(r) for r in aist]);pointsA=np.stack([core_points(r)-core_points(r)[:,:1] for r in aist]);musicA=np.stack([r["music"] for r in aist])
    aux_body=bodyA if arm!="ChangE" else np.stack([z["source139"][i,64:160,7:] for i in (0,1,0,1)])
    aux_points=pointsA if arm!="ChangE" else z["source22"][[0,1,0,1],64:160]-z["source22"][[0,1,0,1],64:160,:1]
    aux_music=musicA if arm!="ChangE" else z["music"][[0,0,1,1]]
    aux_ref=reference(list(aux_body));music=torch.tensor(aux_music,device="cuda");body=torch.tensor(aux_body,device="cuda");points=torch.tensor(aux_points,device="cuda")
    source_cache=root/("models/pilot/change_aux_states.npz" if arm=="ChangE" else "data/aist_states.npz")
    expected_sha=read(root/"models/pilot/result.json")["files"]["change_aux_states.npz"] if arm=="ChangE" else read(root/"data/protocol.json")["files"]["aist_states.npz"]
    assert sha(source_cache)==expected_sha
    states=dict(np.load(source_cache));initial_state_sha=sha(source_cache)
    initial_states={k:v.copy() for k,v in states.items()}
    steps=cfg["pilot_steps"] if arm=="pilot" else cfg["train_steps"];snapshots=[0,1,steps-1] if arm=="pilot" else cfg["snapshot_steps"]
    curves=[];arrays={};history=[];observations=[];actual=[];checks=[];calls=0;newtrajectories=0;refresh_checks=[]
    seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0]
    def curve(update):
        results=[]
        for t in cfg["times"]:
            xt=torch.tensor(initial_states["xt"][initial_states["times"].tolist().index(t)],device="cuda")
            ts=torch.full((4,),t,device="cuda",dtype=torch.long)
            b.clear()
            with torch.no_grad():
                no=d.model(xt,music,None,ts,cond_drop_prob=0);b.set_reference(**aux_ref)
                pred=d.model(xt,music,None,ts,cond_drop_prob=0);b.clear()
                loss,parts=partial_loss(pred,body,points,no,n,s,cfg["partial_weights"],cfg["partial_geometry_beta_m"])
            arrays[f"{update}_{t}"]=pred.cpu().numpy()
            results.append(dict(t=t,loss=float(loss),parts={k:float(v) for k,v in parts.items()}))
        return dict(update=update,records=results,mean_loss=float(np.mean([x["loss"] for x in results])))
    curves.append(curve(0))
    def loss(row):
        if row["task"]=="paired":
            rows=[data[c] for c in row["cases"]];y=n.normalize(torch.tensor(np.stack([r["model_raw"] for r in rows]),device="cuda"))
            m=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
            return paired_objective(d,b,y,m,packet(rows),row["t"],row["seed"])
        xt=torch.tensor(states["xt"][states["times"].tolist().index(row["t"])],device="cuda")
        return partial_objective(d,b,xt,music,aux_ref,row["t"],row["seed"],body,points,n,s,cfg)
    paired_ledger=paired_schedule(ledger,[r["case"] for r in pop if r["domain"]=="aist" and r["role"]=="train"],cfg["seed"])
    schedule=[]
    for i in range(steps):
        if arm!="pilot" and i%2==0:
            row=paired_ledger[i//2];schedule.append(dict(step=i,task="paired",cases=row["cases"],t=row["t"],seed=row["seed"]))
        else:
            j=i if arm=="pilot" else i//2
            schedule.append(dict(step=i,task="partial",domain="aist" if arm!="ChangE" else "change",t=int(states["times"][(j*17)%50]),seed=cfg["seed"]+90000+j,source_count=4 if arm!="ChangE" else 2))
    def refresh(path):
        p,nr,tr=sample(d,b,aux_ref,music,seeds,trace=True,selected=cfg["times"])
        saved=trace_arrays(tr);np.savez_compressed(path,**saved)
        return saved,nr
    if not replay:
        (out/"snapshots").mkdir();checkpoint(b,out/"initial.pt",cfg,"CanonicalIO",0)
        opt=torch.optim.AdamW(b.reference.parameters(),lr=cfg["learning_rate"])
        for row in schedule:
            i=row["step"]
            if arm!="pilot" and i==300:
                checkpoint(b,out/"weights_300.pt",cfg,"CanonicalIO",300);states,nr=refresh(out/"aux_refresh_300.npz");calls+=1;newtrajectories+=4
                refresh_checks.append(dict(update=300,noise=nr,current_reference_digest=digest(b.reference)))
            watch=i in snapshots
            if watch:before=vector(b.reference).clone();checkpoint(b,out/f"snapshots/{i}_pre.pt",cfg,"CanonicalIO",i)
            opt.zero_grad(set_to_none=True);value,parts=loss(row);assert torch.isfinite(value)
            torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            g=train_gradient(b) if watch else None
            torch.nn.utils.clip_grad_norm_(b.reference.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(dict(row,loss=float(value.detach()),parts=parts))
            if watch:
                checkpoint(b,out/f"snapshots/{i}_post.pt",cfg,"CanonicalIO",i+1);observations.append(dict(step=i,loss=float(value.detach()),parts=parts,gradients=g,update_norm=float((vector(b.reference)-before).norm())))
            if (i+1)%100==0:print("R32_TRAIN",arm,i+1,float(value.detach()),flush=True)
        checkpoint(b,out/"adapter.pt",cfg,"CanonicalIO",steps);curves.append(curve(steps));optimizer_steps=steps
    else:
        old=read(Path(replay)/"result.json");assert initial==old["initial_digest"]
        for name,h in old["files"].items():assert sha(Path(replay)/name)==h
        for item in old["observations"]:
            i=item["step"]
            if arm!="pilot" and i>=300:states=dict(np.load(Path(replay)/"aux_refresh_300.npz"))
            restore(b,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(b.reference).clone();b.zero_grad(set_to_none=True)
            value,parts=loss(schedule[i]);torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            g=train_gradient(b);restore(b,Path(replay)/f"snapshots/{i}_post.pt")
            value=dict(step=i,loss=float(value.detach()),parts=parts,gradients=g,update_norm=float((vector(b.reference)-before).norm()))
            checks.append(readout_check(value,item));actual.append(value)
        if arm!="pilot":
            restore(b,Path(replay)/"weights_300.pt");states,nr=refresh(out/"aux_refresh_300.npz");calls+=1;newtrajectories+=4
            same_arrays(out/"aux_refresh_300.npz",Path(replay)/"aux_refresh_300.npz");refresh_checks=old["refresh_checks"]
        restore(b,Path(replay)/"adapter.pt");curves.append(curve(steps));assert curves==old["curves"]
        history=old["history"];observations=old["observations"];optimizer_steps=0
    assert_frozen(d,b,frozen)
    final=digest(b)
    if replay:assert final==old["final_digest"]
    gate=None
    if arm=="pilot":
        music_full=torch.tensor(musicA,device="cuda");pred,nr,_=sample(d,b,reference(list(bodyA)),music_full,seeds);calls+=1;newtrajectories+=4
        raw=n.unnormalize(pred.clone())
        with torch.no_grad():pts=f(raw,s)[:,:,:22]
        baseline=read(Path(cfg["r31"])/"evaluation/result.json")["records"];records=[]
        for i,c in enumerate(cfg["train_cases"]):
            m=metrics(raw[i].cpu().numpy(),pts[i].cpu().numpy(),data[c]["model_raw"],data[c]["model_points"])
            before=next(x for x in baseline if x["arm"]=="CanonicalIO" and x["role"]=="train" and x["mode"]=="correct" and x["seed_index"]==0 and x["case"]==c)["metrics"]
            good=m["body22_rad"]<before["body22_rad"] and .5<=m["body_activity_ratio"]<=1.5 and .7<=m["arm_range_ratio"]<=1.4
            records.append(dict(case=c,metrics=m,baseline=before,improved=good))
        arrays["pilot_final"]=pred.cpu().numpy()
        no,nr2,_=sample(d,b,None,music_full,seeds);calls+=1;newtrajectories+=4
        assert np.array_equal(no.cpu().numpy(),np.load(root/"data/arrays.npz")["train_none_0"]);arrays["pilot_NoRef"]=no.cpu().numpy()
        ratio=curves[-1]["mean_loss"]/curves[0]["mean_loss"];gate=dict(loss_ratio=ratio,records=records,improved_sources=sum(x["improved"] for x in records),NoRef_exact=True)
        gate["passed"]=ratio<=cfg["pilot_loss_ratio_max"] and gate["improved_sources"]>=cfg["pilot_min_improved_sources"]
        if gate["passed"]:
            b.close();d,n,s,f,e,info,b=runtime(cfg)
            p,nr,tr=sample(d,b,reference([z["source139"][i,64:160,7:] for i in (0,1,0,1)]),torch.tensor(z["music"][[0,0,1,1]],device="cuda"),seeds,trace=True,selected=cfg["times"])
            np.savez_compressed(out/"change_aux_states.npz",**trace_arrays(tr));calls+=1;newtrajectories+=4
            assert nr["row_sha256"]==read(Path(cfg["r31"])/"evaluation/result.json")["noise"][0]["noise"]["row_sha256"]
    final_checkpoint_hash=sha((Path(replay) if replay else out)/"adapter.pt")
    np.savez_compressed(out/"curves.npz",**arrays)
    if replay:
        same_arrays(out/"curves.npz",Path(replay)/"curves.npz")
        if gate and gate["passed"]:same_arrays(out/"change_aux_states.npz",Path(replay)/"change_aux_states.npz")
    result=dict(arm=arm,admitted=True,initial_digest=initial,final_digest=final,final_checkpoint_sha256=final_checkpoint_hash,frozen=frozen,optimizer_steps=optimizer_steps,
        reference_parameters=265328,action_io_trainable_parameters=0,curves=curves,history=history,observations=observations,actual_readouts=actual,replay_checks=checks,refresh_checks=refresh_checks,
        initial_aux_state_sha256=initial_state_sha,auxiliary_state_policy="FROZEN_R31_INITIAL; CURRENT_ARM_REFRESH_AT_300_FOR_MAIN; NOT_EVERY_STEP_ON_POLICY",
        partial_source_Root_C4_truth_used=False,partial_full_x0_constructed=False,paired_AIST_truth_used_only_in_paired_task=True,
        sampler_calls=calls,generated_trajectories=newtrajectories,main_training_admitted=bool(gate and gate["passed"]),pilot_gate=gate,
        files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    assert_frozen(d,b,frozen);b.close();write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit()))
    print("R32_TRAIN_DONE",arm,optimizer_steps,"gate",gate["passed"] if gate else None,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--arm",choices=["pilot","AIST","ChangE"],required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.arm,a.replay_from)

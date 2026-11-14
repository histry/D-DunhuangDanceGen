"""Structure first, then matched time/temporal/cache treatments from one selected state."""
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.objective_r19.common import state,parameters,vector,capture,restore
from dunhuang.research.global_clock_r17.train import digest
from dunhuang.research.masked_denoiser.common import tensor_hash
from .common import load,cache_data,selected_digest,frozen_except_null,schedule
from .probes import run as probes
from .training import build,backward,observe_before,observe_after
from .sampler import sample

def source(cfg,root,arm):
    if arm in ("S0","S1"):return ("joint" if arm=="S1" else "split"),None
    gate=read(Path(root)/"structure_gate.json");return gate["layout"],gate["checkpoint"]

def make_cache(cfg,out,cache,predict,step,weight_ref,replay=None):
    bindings=[];values={};folder=out/("cache_"+str(step));folder.mkdir()
    for case in cfg["train_cases"]:
        for hard in (False,True):
            s=state(dict(cfg),cache,case,519,hard,cfg["seed"]+case)
            arrays,receipt=sample(cfg,cache["diffusion"],predict,s,cfg["seed"]+90000+step+case*10,cache,keep_xt=True)
            path=folder/f"{case}_{int(hard)}.npz";np.savez_compressed(path,**arrays)
            if replay:
                old=Path(replay)/folder.name/path.name;same_arrays(path,old)
            for t in receipt["cache_t"]:values[case,hard,t]=torch.tensor(arrays["xt_"+str(t)][None],device="cuda")
            bindings.append(dict(case=case,hard=hard,refresh=step,trace=ref(path),weight=weight_ref,
                source=next(r["left"] for r in cache["rows"] if r["case"]==case),
                data_arrays=next(r["arrays"] for r in cache["rows"] if r["case"]==case),
                music_sha256=tensor_hash(s[1]),scaler=cache["assets"]["scaler"],
                condition_sha256=tensor_hash(s[3]),known_sha256=tensor_hash(s[4]),origin=cache["cache"][case][0]["origin"].tolist(),receipt=receipt))
    return values,bindings

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out);layout,start_ref=source(cfg,root,arm)
    if replay:forbid_optimizers()
    w,d,data,predict,head=load(cfg,layout,start_ref);cache=cache_data(cfg,w,d,data)
    oldweights=read(Path(cfg["parent"])/"calibration/result.json")["weights"]
    extra_ref=ref(root/"temporal_calibration/result.json") if arm in ("A2","A3") else None
    extra=read(checked(extra_ref))["weights"] if extra_ref else None
    initial=dict(selected=selected_digest(w),head=digest(head),frozen=w.digest(True),frozen_except_null=frozen_except_null(w))
    initial_q,initial_scores=probes(cfg,cache,predict,full=False);np.savez_compressed(out/"initial_q.npz",**initial_q)
    rows=schedule(cfg,arm);history=[];updates=[];cache_bindings=[];cache_values={};torch.cuda.reset_peak_memory_stats();wall=time.perf_counter()
    if not replay:
        (out/"snapshots").mkdir();optimizer=torch.optim.AdamW(parameters(w,head),lr=cfg["learning_rate"])
        for row in rows:
            step=row["step"]
            if arm in ("A2","A3") and step in cfg["cache_refresh"]:
                weight=start_ref if step==0 else updates[-1]["post"]
                before=(w.digest(),digest(head))
                cache_values,bindings=make_cache(cfg,out,cache,predict,step,weight);cache_bindings+=bindings
                assert before==(w.digest(),digest(head))
            s=list(state(cfg,cache,row["case"],row["t"],row["hard"],row["seed"]))
            if row["use_cache"]:s[-1]=cache_values[row["case"],row["hard"],row["t"]]
            row.update(x_t_sha256=tensor_hash(s[-1]),known_sha256=tensor_hash(s[4]))
            watch=step in cfg["snapshot_steps"];q=predict.telemetry;q.enabled=watch;q.grad_enabled=watch
            pred,terms,counts,active,objective=build(cfg,predict,cache,s,oldweights,extra)
            if not torch.isfinite(objective):raise ValueError("Nonfinite training objective")
            if watch:
                pre=capture(cfg,w,head,out/"snapshots"/f"{step}_pre.pt",arm,step)
                before=vector(w,head);record,gu=observe_before(w,head,pred,terms,counts,active,cache)
                forward=copy.deepcopy(q.forward);q.backward={}
            clipped=backward(w,head,objective)
            if watch:back=copy.deepcopy(q.backward)
            optimizer.step();q.enabled=False;q.grad_enabled=False
            if watch:
                delta=vector(w,head)-before
                observed=observe_after(record,gu,delta,predict,s,w,head,forward,back,clipped)
                post=capture(cfg,w,head,out/"snapshots"/f"{step}_post.pt",arm,step+1)
                updates.append(dict(step=step,pre=pre,post=post,readout=observed))
            history.append(dict(step=step,unknown=float(terms["unknown"].detach()),loss=float(objective.detach()),
                motion_gradient=clipped["motion_norm"],head_gradient=clipped["head_norm"]))
            if (step+1)%40==0:print("R20_TRAIN",arm,layout,step+1,history[-1],flush=True)
        checkpoint=updates[-1]["post"];optimizer_steps=cfg["steps"]
    else:
        old=read(Path(replay)/"result.json")
        assert initial==old["initial"] and initial_scores==old["initial_scores"]
        same_arrays(out/"initial_q.npz",Path(replay)/"initial_q.npz")
        # Rebuild own-policy caches from exact saved weight snapshots, never optimize.
        for refresh in cfg["cache_refresh"] if arm in ("A2","A3") else []:
            bindings=[b for b in old["cache_bindings"] if b["refresh"]==refresh]
            restore(cfg,w,head,bindings[0]["weight"])
            _,new=make_cache(cfg,out,cache,predict,refresh,bindings[0]["weight"],replay)
            for a,b in zip(bindings,new):assert a["receipt"]==b["receipt"]
        for item in old["update_observations"]:
            row=rows[item["step"]];s=list(state(cfg,cache,row["case"],row["t"],row["hard"],row["seed"]))
            if row["use_cache"]:
                refresh=max(x for x in cfg["cache_refresh"] if x<=row["step"])
                b=next(b for b in old["cache_bindings"] if b["refresh"]==refresh and b["case"]==row["case"] and b["hard"]==row["hard"])
                s[-1]=torch.tensor(np.load(checked(b["trace"]))["xt_"+str(row["t"])][None],device="cuda")
            restore(cfg,w,head,item["pre"]);q=predict.telemetry;q.enabled=True;q.grad_enabled=True
            pred,terms,counts,active,objective=build(cfg,predict,cache,s,oldweights,extra)
            before=vector(w,head);record,gu=observe_before(w,head,pred,terms,counts,active,cache)
            forward=copy.deepcopy(q.forward);q.backward={}
            clipped=backward(w,head,objective);back=copy.deepcopy(q.backward);q.enabled=False;q.grad_enabled=False
            restore(cfg,w,head,item["post"]);delta=vector(w,head)-before
            observed=observe_after(record,gu,delta,predict,s,w,head,forward,back,clipped)
            assert observed==item["readout"],"Actual-update/layer replay differs"
        checkpoint=old["checkpoint"];restore(cfg,w,head,checkpoint);optimizer_steps=0;updates=old["update_observations"];cache_bindings=old["cache_bindings"]
        for row in rows:
            s=list(state(cfg,cache,row["case"],row["t"],row["hard"],row["seed"]))
            if row["use_cache"]:
                refresh=max(x for x in cfg["cache_refresh"] if x<=row["step"])
                b=next(b for b in cache_bindings if b["refresh"]==refresh and b["case"]==row["case"] and b["hard"]==row["hard"])
                s[-1]=torch.tensor(np.load(checked(b["trace"]))["xt_"+str(row["t"])][None],device="cuda")
            row.update(x_t_sha256=tensor_hash(s[-1]),known_sha256=tensor_hash(s[4]))
        assert rows==old["ledger"]
    q,scores=probes(cfg,cache,predict);np.savez_compressed(out/"q_eval.npz",**q)
    assert w.digest(True)==initial["frozen"] and frozen_except_null(w)==initial["frozen_except_null"]
    result=dict(arm=arm,layout=layout,checkpoint=checkpoint,initial=initial,final_selected=selected_digest(w),final_head=digest(head),
        initial_scores=initial_scores,scores=scores,ledger=rows,history=history,update_observations=updates,
        cache_bindings=cache_bindings,optimizer_steps=optimizer_steps,
        motion_parameters=sum(p.numel() for p in w.selected()),head_parameters=sum(p.numel() for p in head.parameters()),
        null_adaptation=getattr(w,"null_adaptation",None),wall_seconds=time.perf_counter()-wall,
        peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),predictor_forward_calls=predict.calls,
        motion_token_evaluations=predict.calls*2*(225 if layout=="joint" else 300),
        whole_song_generations=0,operational_qualified=False)
    if replay:
        assert scores==old["scores"] and result["final_selected"]==old["final_selected"] and result["final_head"]==old["final_head"]
        same_arrays(out/"q_eval.npz",Path(replay)/"q_eval.npz")
    if extra_ref:checked(extra_ref)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,source=start_ref,
        initialization="Same R19 N2" if start_ref is None else "Same selected structural endpoint; fresh AdamW",
        layout=layout,temporal_weights=extra_ref,task="inbetween",evaluation_used_for_training=False))
    predict.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--arm",choices=["S0","S1","A0","A1","A2","A3"],required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.arm,a.replay_from)

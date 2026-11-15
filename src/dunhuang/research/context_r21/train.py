"""Fixed-budget 2x2 and a shared continuation control for cache/support ablations."""
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.objective_r19.common import capture,restore,parameters,vector
from dunhuang.research.temporal_r20.common import selected_digest,frozen_except_null
from dunhuang.research.temporal_r20.training import backward,observe_before,observe_after
from dunhuang.research.global_clock_r17.train import digest
from dunhuang.research.masked_denoiser.common import tensor_hash
from .engine import load,cache_data,state,schedule,source,layout,balanced,weights,probes,branch_readout
from .objective import build
from .sampler import sample

def make_cache(cfg,out,cache,predict,step,weight_ref,replay=None):
    folder=out/("cache_"+str(step));folder.mkdir();values={};bindings=[]
    for case in cache["population"]["follow_cases"]:
        s,_,_=state(cfg,cache,case,519)
        arrays,receipt=sample(cfg,cache,predict,s,cfg["seed"]+90000+step+case,keep_xt=True)
        path=folder/(str(case)+".npz");np.savez_compressed(path,**arrays)
        if replay:
            same_arrays(path,Path(replay)/folder.name/path.name)
        for t in receipt["cache_t"]:values[case,t]=torch.tensor(arrays["xt_"+str(t)][None],device="cuda")
        row=next(r for r in cache["rows"] if r["case"]==case)
        bindings.append(dict(case=case,refresh=step,trace=ref(path),weight=weight_ref,source=row["left"],
            data_arrays=row["arrays"],scaler=cache["assets"]["scaler"],music_sha256=tensor_hash(s[1]),
            origin=cache["cache"][case][0]["origin"].tolist(),receipt=receipt))
    return values,bindings

def input_state(cfg,cache,row,values=None,bindings=None):
    s,_,proxy=state(cfg,cache,row["case"],row["t"],row["variant"],row["pattern"],row["seed"]);s=list(s)
    if row["use_cache"]:
        if values is not None:s[-1]=values[row["case"],row["t"]]
        else:
            refresh=max(b["refresh"] for b in bindings if b["refresh"]<=row["step"])
            b=next(b for b in bindings if b["refresh"]==refresh and b["case"]==row["case"])
            with np.load(checked(b["trace"])) as z:s[-1]=torch.tensor(z["xt_"+str(row["t"])][None],device="cuda")
    return s,proxy

def model_meta(w,h):
    return dict(selected=selected_digest(w),head=digest(h),frozen=w.digest(True),frozen_except_null=frozen_except_null(w))

def run(cfg,root,out,arm,replay=None):
    out=fresh(out);root=Path(root);follow=arm in ("Q","M","F")
    if arm=="F":assert read(root/"data/result.json")["support_admitted"]
    if replay:forbid_optimizers()
    w,d,data,p,h=load(cfg,root,arm);cache=cache_data(cfg,root,d,data);old_weights,temp_weights=weights(cfg)
    support_weights=read(root/"support_calibration/result.json")["weights"] if arm=="F" else None
    initial=model_meta(w,h);rows=schedule(cfg,cache["population"],arm);history=[];updates=[];curves=[];bindings=[];values={};curve_refs={}
    points=cfg["follow_curve_updates"] if follow else cfg["curve_updates"]
    watch_steps=cfg["follow_snapshot_steps"] if follow else cfg["snapshot_steps"]
    if not replay:
        (out/"snapshots").mkdir()
        initial_ref=capture(cfg,w,h,out/"snapshots/0_pre.pt",arm,0)
    else:
        previous=read(Path(replay)/"result.json");initial_ref=previous["initial_checkpoint"]
        assert initial==previous["initial"];restore(cfg,w,h,initial_ref)
    def curve(update,weight):
        scores,arrays=probes(cfg,cache,p);path=out/("curve_"+str(update)+".npz");np.savez_compressed(path,**arrays)
        result=dict(update=update,weight=weight,scores=scores,predictions=ref(path))
        if replay:
            old=next(x for x in previous["curves"] if x["update"]==update)
            assert scores==old["scores"];same_arrays(path,checked(old["predictions"]))
        curves.append(result)
    curve(0,initial_ref);wall=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if not replay:
        optimizer=torch.optim.AdamW(parameters(w,h),lr=cfg["learning_rate"])
        for row in rows:
            step=row["step"]
            if follow and step in cfg["cache_refresh"]:
                weight=source(cfg,root,arm) if step==0 else curve_refs[step]
                before=model_meta(w,h);values,bs=make_cache(cfg,out,cache,p,step,weight);bindings+=bs;assert before==model_meta(w,h)
            s,proxy=input_state(cfg,cache,row,values=values)
            row.update(x_t_sha256=tensor_hash(s[-1]),known_sha256=tensor_hash(s[4]))
            watch=step in watch_steps;p.telemetry.enabled=watch;p.telemetry.grad_enabled=watch;p.capture_branches=watch
            pred,terms,counts,active,obj=build(cfg,p,cache,s,old_weights,temp_weights,support_weights,balanced(arm),proxy if arm=="F" else None)
            assert torch.isfinite(obj)
            if watch:
                pre=initial_ref if step==0 else capture(cfg,w,h,out/"snapshots"/(str(step)+"_pre.pt"),arm,step)
                before=vector(w,h);record,gu=observe_before(w,h,pred,terms,counts,active,cache)
                branches=branch_readout(p,s[0],s[4],w,h);forward=copy.deepcopy(p.telemetry.forward);p.telemetry.backward={}
            clipping=backward(w,h,obj)
            if watch:back=copy.deepcopy(p.telemetry.backward)
            optimizer.step()
            if watch:
                actual=observe_after(record,gu,vector(w,h)-before,p,s,w,h,forward,back,clipping)
                actual.update(branches_before=branches,branches_after=branch_readout(p,s[0],s[4]))
                post=capture(cfg,w,h,out/"snapshots"/(str(step)+"_post.pt"),arm,step+1)
                updates.append(dict(step=step,pre=pre,post=post,readout=actual))
            p.telemetry.enabled=False;p.telemetry.grad_enabled=False;p.capture_branches=False
            history.append(dict(step=step,unknown=float(terms["unknown"].detach()),loss=float(obj.detach())))
            if step+1 in points:
                post=updates[-1]["post"] if watch else capture(cfg,w,h,out/"snapshots"/(str(step)+"_post.pt"),arm,step+1)
                curve_refs[step+1]=post;curve(step+1,post)
            if (step+1)%100==0:print("R21_TRAIN",arm,step+1,history[-1],flush=True)
        final=curve_refs[len(rows)];optimizer_steps=len(rows)
    else:
        for refresh in cfg["cache_refresh"] if follow else []:
            old=[b for b in previous["cache_bindings"] if b["refresh"]==refresh];restore(cfg,w,h,old[0]["weight"])
            _,new=make_cache(cfg,out,cache,p,refresh,old[0]["weight"],replay)
            assert [b["receipt"] for b in new]==[b["receipt"] for b in old]
        bindings=previous["cache_bindings"]
        for item in previous["update_observations"]:
            row=rows[item["step"]];s,proxy=input_state(cfg,cache,row,bindings=bindings);restore(cfg,w,h,item["pre"])
            p.telemetry.enabled=True;p.telemetry.grad_enabled=True;p.capture_branches=True
            pred,terms,counts,active,obj=build(cfg,p,cache,s,old_weights,temp_weights,support_weights,balanced(arm),proxy if arm=="F" else None)
            before=vector(w,h);record,gu=observe_before(w,h,pred,terms,counts,active,cache)
            branches=branch_readout(p,s[0],s[4],w,h);forward=copy.deepcopy(p.telemetry.forward);p.telemetry.backward={}
            clipping=backward(w,h,obj);back=copy.deepcopy(p.telemetry.backward)
            restore(cfg,w,h,item["post"])
            actual=observe_after(record,gu,vector(w,h)-before,p,s,w,h,forward,back,clipping)
            actual.update(branches_before=branches,branches_after=branch_readout(p,s[0],s[4]))
            assert actual==item["readout"],("Update replay",arm,item["step"])
            p.telemetry.enabled=False;p.telemetry.grad_enabled=False;p.capture_branches=False
        for row in rows:
            s,_=input_state(cfg,cache,row,bindings=bindings);row.update(x_t_sha256=tensor_hash(s[-1]),known_sha256=tensor_hash(s[4]))
        assert rows==previous["ledger"]
        for c in previous["curves"][1:]:
            restore(cfg,w,h,c["weight"]);curve(c["update"],c["weight"])
        final=previous["checkpoint"];restore(cfg,w,h,final);optimizer_steps=0;updates=previous["update_observations"]
    final_scores,arrays=probes(cfg,cache,p,controls=True,dev=False);np.savez_compressed(out/"final_q.npz",**arrays)
    # DEV is evaluated only after all branches are trained by a separate final phase below.
    # Training-time probes above must remain Train-only.
    assert w.digest(True)==initial["frozen"] and frozen_except_null(w)==initial["frozen_except_null"]
    result=dict(arm=arm,layout=layout(arm),balanced=balanced(arm),initial=initial,initial_checkpoint=initial_ref,checkpoint=final,
        curves=curves,ledger=rows,history=history,update_observations=updates,cache_bindings=bindings,final_scores=final_scores,
        final_selected=selected_digest(w),final_head=digest(h),optimizer_steps=optimizer_steps,
        trainable_motion=sum(x.numel() for x in w.selected()),trainable_head=sum(x.numel() for x in h.parameters()),
        wall_seconds=time.perf_counter()-wall,peak_cuda_bytes=torch.cuda.max_memory_allocated(),forward_calls=p.calls,
        branch_training="guided output only; branch losses are readouts",whole_song_generations=0,operational_qualified=False)
    if replay:
        assert result["final_scores"]==previous["final_scores"];same_arrays(out/"final_q.npz",Path(replay)/"final_q.npz")
        assert result["final_selected"]==previous["final_selected"] and result["final_head"]==previous["final_head"]
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,source=source(cfg,root,arm),data=ref(root/"data/result.json"),
        dev_selection=False,objective="A2 pooled or independently normalized crossing/editable means",final_checkpoint_only=True))
    p.close()
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--config",required=True);a.add_argument("--root",required=True);a.add_argument("--out",required=True);a.add_argument("--arm",required=True);a.add_argument("--replay-from")
    x=a.parse_args();run(read(x.config),x.root,x.out,x.arm,x.replay_from)
